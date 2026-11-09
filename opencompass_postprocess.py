"""Option extractor for multiple-choice utility benchmarks (PIQA).

Originally copied verbatim from OpenCompass 0.5.4,
`opencompass/utils/text_postprocessors.py` (Apache License 2.0,
https://github.com/open-compass/opencompass).

Our manual audit of all 2,176 generated utility responses uncovered four bugs in
OpenCompass 0.5.4's `first_option_postprocess` when applied to English
instruction-tuned / safety-realigned models:

1. Missing comma after `rf'故选?\\s*([{options}])'` (line 35 in OpenCompass 0.5.4;
   reported upstream in OpenCompass Issue #2704 / PR #2705), which implicitly
   concatenates two adjacent regex strings.
2. Wrong capture group in `rf'(\\s|^)[{options}][\\s。，,：:\\.$]'` (line 70):
   parentheses wrap `(\\s|^)` (leading whitespace) instead of `([{options}])`.
   When an English reply says `"follow option A."` or `"Option A seems..."`,
   `match.group(1)` is `' '` (a non-empty space), so `outputs = ' '` and the
   option loop `for i in options: if i in ' '` fails.
3. Unbounded fallback / cushion (`rf'1.\\s?(.*?)$'` and `rf'([{options}])'`):
   after (2) fails, the extractor scans left-to-right for ANY uppercase `A` or
   `B` anywhere in the text:
   - False negatives: `"To make Skor Bar Ice Cream at home, you can follow
     option A"` extracts `'B'` from `"Skor Bar"` because `'B'` appears earlier
     in the sentence than `"option A"`. Similarly, `"A draw/erase board is...
     Therefore, the correct answer would be B"` extracts `'A'` from the leading
     indefinite article `"A"`.
   - False positives on refusals & `"Neither A nor B"` (phantom accuracy):
     Safety-realigned models frequently refuse or reject both choices with
     `"Neither option A nor B is correct"`, `"I would strongly advise against
     both options A and B"`, or `"I cannot provide instructions on how to make
     a bomb. Burning..."`. The cushion regex extracts `'A'` or `'B'` from those
     refusals, awarding up to +25% phantom accuracy on PIQA whenever `gold=='A'`.

`first_option_postprocess_opencompass_054` preserves the verbatim OpenCompass
0.5.4 implementation for reference; `first_option_postprocess` implements the
corrected extractor.
"""
import re


def first_option_postprocess(text: str, options: str = "AB", cushion: bool = True) -> str:
    """Extract the selected multiple-choice option without matching refusals,
    'Neither A nor B' abstentions, or incidental capital letters in prose."""
    t = str(text).strip()
    if not t:
        return ""
    opts = re.escape(options)

    # 1. Explicit conclusion ("The correct answer is: **A", "Answer: B",
    #    "the correct option would be B", etc.). Standalone option letter must
    #    be uppercase so the English article "a" ("The correct answer is a
    #    combination of...") is never mistaken for option A.
    explicit_conclusion = re.search(
        rf"(?i)(?:the\s+(?:correct|best|right)\s+(?:answer|option)\s+(?:is|would\s+be|appears\s+to\s+be)\s*:?\s*\*?\*?|(?:\*\*|\b)answer\s*:\s*\*?\*?)\s*(?:option\s*)?\(?((?-i:[{opts}]))\b",
        t,
    )
    if explicit_conclusion:
        after = t[explicit_conclusion.start():explicit_conclusion.end() + 25].lower()
        if not any(neg in after for neg in ["neither", "not a or b", "not to attempt", "no option"]):
            return explicit_conclusion.group(1).upper()

    # 2. Rejections of both options / "neither" / safety refusals of the options:
    if re.search(
        r"(?i)(?:^\s*neither\b|\bneither\s+(?:option\s+)?A\s+(?:nor|or)\s+(?:option\s+)?B\b|\bneither\s+of\s+(?:the|these|those|the\s+above)\s+(?:options?|statements?|choices?)\b|\b(?:recommend|say)\s+neither\b|\badvise\s+against\s+both\b|\badvise\s+against\s+option\s+A\b|\bboth\s+options\s+A\s+and\s+B\s+are\b|\boptions\s+A\s+and\s+B\s+are\s+creative\s+ideas\b|\bfollow\s+either\s+option\s+A\s+or\s+B\b)",
        t,
    ):
        return ""

    # 3. Direct option selections ("A. ...", "Option A is/seems/would be...",
    #    "I would recommend option B", "you can follow option A", etc.):
    pos_pats = [
        rf"^([{opts}])\.",
        rf"^([{opts}])\s+(?i:is\s+(?:the\s+)?(?:correct|better|best)\s+answer)\b",
        rf"(?i)\bI\s+would\s+(?:highly\s+|strongly\s+)?(?:recommend|advise|suggest|choose|say(?:\s+that)?)\s+(?:option\s+|answer\s+)?((?-i:[{opts}]))\b",
        rf"(?i)\bOption\s+((?-i:[{opts}]))\s+(?:is|seems\s+to\s+be|would\s+be)\s+(?:the\s+|a\s+)?(?:correct|better|best|most\s+appropriate|more\s+accurate)\b",
        rf"(?i)\b(?:you\s+(?:can|should|would)\s+(?:follow|use|choose)|is\s+the\s+correct\s+option|would\s+be)\s+option\s+((?-i:[{opts}]))\b",
    ]
    for pat in pos_pats:
        m = re.search(pat, t)
        if m:
            return m.group(1).upper()

    # 4. Standalone option line after a brief intro (only when a single option
    #    line is given and it does not negate the option with ", not a "):
    line_opts = re.findall(rf"(?:^|\n)\s*([{opts}])\.\s+([^\n]+)", t)
    if len(line_opts) == 1 and ", not a " not in line_opts[0][1].lower():
        return line_opts[0][0].upper()
    return ""


def first_option_postprocess_opencompass_054(text: str, options: str, cushion=True) -> str:
    """Verbatim OpenCompass 0.5.4 implementation (kept for audit reference)."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        rf'答案是?\s*([{options}])',
        rf'答案是?\s*：\s*([{options}])',
        rf'答案是?\s*:\s*([{options}])',
        rf'答案选项应?该?是\s*([{options}])',
        rf'答案选项应?该?为\s*([{options}])',
        rf'答案应该?是\s*([{options}])',
        rf'答案应该?选\s*([{options}])',
        rf'答案选项为?\s*：\s*([{options}])',
        rf'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'答案选项是?\s*:\s*([{options}])',
        rf'答案为\s*([{options}])',
        rf'答案选\s*([{options}])',
        rf'选择?\s*([{options}])',
        rf'故选?\s*([{options}])'
        rf'只有选?项?\s?([{options}])\s?是?对',
        rf'只有选?项?\s?([{options}])\s?是?错',
        rf'只有选?项?\s?([{options}])\s?不?正确',
        rf'只有选?项?\s?([{options}])\s?错误',
        rf'说法不?对选?项?的?是\s?([{options}])',
        rf'说法不?正确选?项?的?是\s?([{options}])',
        rf'说法错误选?项?的?是\s?([{options}])',
        rf'([{options}])\s?是正确的',
        rf'([{options}])\s?是正确答案',
        rf'选项\s?([{options}])\s?正确',
        rf'所以答\s?([{options}])',
        rf'所以\s?([{options}][.。$]?$)',
        rf'所有\s?([{options}][.。$]?$)',
        rf'[\s，：:,]([{options}])[。，,\.]?$',
        rf'[\s，,：:][故即]([{options}])[。\.]?$',
        rf'[\s，,：:]因此([{options}])[。\.]?$',
        rf'[是为。]\s?([{options}])[。\.]?$',
        rf'因此\s?([{options}])[。\.]?$',
        rf'显然\s?([{options}])[。\.]?$',
        rf'答案是\s?(\S+)(?:。|$)',
        rf'答案应该是\s?(\S+)(?:。|$)',
        rf'答案为\s?(\S+)(?:。|$)',
        rf'(?i)ANSWER\s*:\s*([{options}])',
        rf'[Tt]he answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        rf'^选项\s?([{options}])',
        rf'^([{options}])\s?选?项',
        rf'(\s|^)[{options}][\s。，,：:\.$]',
        rf'1.\s?(.*?)$',
        rf'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        rf'([{options}]):',
        rf'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''
