from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from loraeval.data import (
    LABEL_NAMES,
    LABEL_TO_ID,
    Collator,
    Example,
    TextClassificationDataset,
    label_distribution,
    make_loader,
    parse_phrasebank,
    read_phrasebank_zip,
    stratified_split,
)


class TestParsing:
    def test_parse_lines(self) -> None:
        lines = ["Profit rose .@positive", "", "  Sales fell@negative  ", "Board met@NEUTRAL"]
        ex = parse_phrasebank(lines)
        assert ex == [
            Example("Profit rose .", LABEL_TO_ID["positive"]),
            Example("Sales fell", LABEL_TO_ID["negative"]),
            Example("Board met", LABEL_TO_ID["neutral"]),
        ]

    def test_email_like_at_signs_are_handled_by_rsplit(self) -> None:
        assert (
            parse_phrasebank(["contact a@b.com today@neutral"])[0].text == "contact a@b.com today"
        )

    @pytest.mark.parametrize("line", ["no separator", "text@bogus"])
    def test_malformed_lines_raise(self, line: str) -> None:
        with pytest.raises(ValueError, match="line 1"):
            parse_phrasebank([line])

    def test_read_zip_is_latin1_and_skips_macosx(self, tmp_path: Path) -> None:
        content = "Señor profit rose .@positive\nloss@negative\n".encode("latin-1")
        zip_path = tmp_path / "fpb.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", content)
            zf.writestr("__MACOSX/FinancialPhraseBank-v1.0/._Sentences_AllAgree.txt", b"junk")
        ex = read_phrasebank_zip(zip_path, "all")
        assert len(ex) == 2 and ex[0].text.startswith("Señor")
        with pytest.raises(FileNotFoundError):
            read_phrasebank_zip(zip_path, "75")
        with pytest.raises(ValueError):
            read_phrasebank_zip(zip_path, "99")  # type: ignore[arg-type]


class TestSplit:
    def test_stratified_proportions_and_disjointness(
        self, synthetic_examples: list[Example]
    ) -> None:
        split = stratified_split(synthetic_examples, val_fraction=0.15, test_fraction=0.15, seed=0)
        sizes = split.sizes()
        assert sum(sizes.values()) == len(synthetic_examples)
        assert sizes["test"] == pytest.approx(0.15 * len(synthetic_examples), abs=2)
        total = label_distribution(synthetic_examples)
        for part in (split.train, split.val, split.test):
            dist = label_distribution(part)
            for name in LABEL_NAMES:
                assert dist[name] / len(part) == pytest.approx(
                    total[name] / len(synthetic_examples), abs=0.02
                )
        texts = [{e.text for e in part} for part in (split.train, split.val, split.test)]
        assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])

    def test_deterministic_and_seed_sensitive(self, synthetic_examples: list[Example]) -> None:
        a = stratified_split(synthetic_examples, val_fraction=0.1, test_fraction=0.1, seed=1)
        b = stratified_split(synthetic_examples, val_fraction=0.1, test_fraction=0.1, seed=1)
        c = stratified_split(synthetic_examples, val_fraction=0.1, test_fraction=0.1, seed=2)
        assert a.test == b.test
        assert a.test != c.test

    @pytest.mark.parametrize(("val", "test"), [(-0.1, 0.1), (0.5, 0.5), (0.0, 1.0)])
    def test_invalid_fractions(self, val: float, test: float) -> None:
        with pytest.raises(ValueError):
            stratified_split([Example("x", 0)], val_fraction=val, test_fraction=test, seed=0)


class TestBatching:
    def test_collator_shapes(self, fake_tokenizer: PreTrainedTokenizerFast) -> None:
        batch = [Example("profit rose", 2), Example("loss fell down weak", 0)]
        out = Collator(fake_tokenizer, max_length=16)(batch)
        assert out["input_ids"].shape == out["attention_mask"].shape == (2, 6)  # CLS + 4 + SEP
        assert out["labels"].tolist() == [2, 0]
        assert out["attention_mask"][0].sum() == 4  # CLS profit rose SEP
        with pytest.raises(ValueError):
            Collator(fake_tokenizer, max_length=0)

    def test_truncation(self, fake_tokenizer: PreTrainedTokenizerFast) -> None:
        out = Collator(fake_tokenizer, max_length=4)(
            [Example("profit rose surge growth record", 2)]
        )
        assert out["input_ids"].shape[1] == 4

    def test_loader_is_seeded(
        self, fake_tokenizer: PreTrainedTokenizerFast, synthetic_examples: list[Example]
    ) -> None:
        ds = TextClassificationDataset(synthetic_examples[:10])
        assert len(ds) == 10 and ds[0] == synthetic_examples[0]
        first = [
            next(
                iter(
                    make_loader(
                        synthetic_examples[:10],
                        fake_tokenizer,
                        batch_size=4,
                        max_length=16,
                        shuffle=True,
                        seed=3,
                    )
                )
            )["labels"]
            for _ in range(2)
        ]
        assert torch.equal(first[0], first[1])
        n = sum(
            len(b["labels"])
            for b in make_loader(
                synthetic_examples[:10], fake_tokenizer, batch_size=4, max_length=16, shuffle=False
            )
        )
        assert n == 10
