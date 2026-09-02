"""
Reproduce the FDA label LLM classification step used in the study.

Input
-----
Either:
    selected_xml/
or:
    selected_xml.zip

The input should contain the 2,836 curated FDA SPL XML files.

Pipeline
--------
1. Stage 1: extract verbatim evidence relevant to placental transfer.
2. Stage 2: classify each label as crosses / no_cross / unknown
   using only the Stage 1 evidence.
3. Exclude unknown labels.
4. Save the retained non-unknown labels.

Historical study result
-----------------------
2,836 curated FDA labels -> 610 retained non-unknown labels.

The later expansion from 610 labels to 802 ingredient-level records is
not performed here; that step was handled separately in the study.

API
---
Uses the Groq API with:
    model = llama-3.3-70b-versatile
    temperature = 0
    max_tokens = 4096

The API key is never stored in the script. Set GROQ_API_KEY in the
environment or enter it securely when prompted.
"""

# Required packages:
#   pip install langchain-core langchain-groq tqdm
#
# Python standard-library modules handle XML/ZIP/CSV processing.
# ================== 0. Environment and imports ==================
import os
import sys
import time
import json
import csv
import random
import zipfile
import getpass
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

# Use tqdm when available; otherwise fall back to a normal iterator.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ---------- Repository-relative paths ----------
CURRENT_DIR = Path.cwd().resolve()

def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / "selected_xml").exists()
        or (path / "selected_xml.zip").exists()
    )

if _looks_like_project_root(CURRENT_DIR):
    PROJECT_ROOT = CURRENT_DIR
elif _looks_like_project_root(CURRENT_DIR.parent):
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    raise FileNotFoundError(
        "Repository root could not be identified. "
        "Expected 'selected_xml/' or 'selected_xml.zip' in the current "
        "directory or its parent directory."
    )

SELECTED_XML_DIR = PROJECT_ROOT / "selected_xml"
SELECTED_XML_ZIP = PROJECT_ROOT / "selected_xml.zip"
EXTRACTED_XML_DIR = PROJECT_ROOT / ".selected_xml_extracted"

if SELECTED_XML_DIR.exists():
    ROOT_DIR = str(SELECTED_XML_DIR)

elif SELECTED_XML_ZIP.exists():
    # A zip archive is convenient for GitHub distribution. Extract it
    # locally on first use; the extracted directory can be deleted at any time.
    if not EXTRACTED_XML_DIR.exists():
        print(
            f"Extracting {SELECTED_XML_ZIP.name} "
            f"to {EXTRACTED_XML_DIR} ..."
        )
        EXTRACTED_XML_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        with zipfile.ZipFile(
            SELECTED_XML_ZIP,
            "r",
        ) as archive:
            archive.extractall(
                EXTRACTED_XML_DIR
            )

    ROOT_DIR = str(EXTRACTED_XML_DIR)

else:
    raise FileNotFoundError(
        "Neither selected_xml/ nor selected_xml.zip was found."
    )

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "02_llm_fda_labeling"
)

LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Stage 1: evidence extraction
PLACENTA_OUTPUT_CSV = str(
    OUTPUT_DIR
    / "llm_stage1_evidence.csv"
)

PROCESSED_LOG_PATH = str(
    LOG_DIR
    / "stage1_processed_files.log"
)

ERROR_LOG_PATH = str(
    LOG_DIR
    / "stage1_error_files.log"
)

# Stage 2: judgement
FIRST_STAGE_CSV = PLACENTA_OUTPUT_CSV

SECOND_STAGE_CSV = str(
    OUTPUT_DIR
    / "llm_stage2_judgements.csv"
)

JUDGE_PROCESSED_LOG = str(
    LOG_DIR
    / "stage2_processed_rows.log"
)

JUDGE_ERROR_LOG = str(
    LOG_DIR
    / "stage2_error_rows.log"
)

# Final non-unknown output
RETAINED_OUTPUT_CSV = str(
    OUTPUT_DIR
    / "llm_retained_non_unknown_results.csv"
)

SUMMARY_JSON = str(
    OUTPUT_DIR
    / "llm_reproduction_summary.json"
)

SETTINGS_JSON = str(
    OUTPUT_DIR
    / "llm_reproduction_settings.json"
)

# ---------- Reproduction settings ----------
# 0 = no batch-size limit.
# If desired for a trial run, change these to a small number such as 10.
MAX_FIRST_STAGE_NEW_FILES = 0
MAX_SECOND_STAGE_NEW_ROWS = 0

PER_FILE_SLEEP_SECONDS = 0
PER_ROW_SLEEP_SECONDS = 0

EXPECTED_XML_COUNT = 2836
HISTORICAL_RETAINED_COUNT = 610

# XML namespace (SPL)
NS = {"hl7": "urn:hl7-org:v3"}


# ---------- 일일 한도용 커스텀 예외 ----------
class DailyLimitReached(Exception):
    """일일 토큰/레이트 한도에 도달했을 때 전체 루프를 멈추기 위한 예외."""
    pass


# ================== 1. LLM 설정 ==================
# Use an existing environment variable if available.
# Otherwise, prompt securely at runtime. The key is never written to disk.
groq_api_key = os.getenv(
    "GROQ_API_KEY",
    "",
).strip().strip('"').strip("'")

if not groq_api_key:
    groq_api_key = getpass.getpass(
        "Enter GROQ_API_KEY: "
    ).strip().strip('"').strip("'")

if not groq_api_key:
    raise ValueError(
        "A GROQ_API_KEY is required to run the LLM pipeline."
    )

os.environ[
    "GROQ_API_KEY"
] = groq_api_key

print(
    "GROQ API key loaded successfully "
    "(key value is not displayed or saved)."
)

# Exact model settings used in the original analysis.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=4096,
)


# ================== 1-1. rate-limit(429) 백오프 유틸 ==================
def call_with_backoff(fn, *args, max_retries=5, base_delay=10, **kwargs):
    """
    rate-limit(429) 또는 'rate limit' 에러 시 지수 백오프로 재시도.
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = repr(e).lower()
            if "429" in msg or "rate limit" in msg or "too many requests" in msg:
                if attempt == max_retries - 1:
                    raise DailyLimitReached(
                        f"LLM rate limit로 {max_retries}회 재시도 후 실패"
                    ) from e

                delay = base_delay * (2 ** attempt) + random.uniform(0, 3)
                print(
                    f"\n⚠️ rate-limit 감지 ({attempt+1}/{max_retries}차 시도 실패). "
                    f"{delay:.1f}초 대기 후 재시도합니다..."
                )
                time.sleep(delay)
                continue
            else:
                raise


# ================== 2. SPL XML 파싱 유틸 ==================
def load_spl_root(path: str):
    """
    path가 .xml이면 그대로 파싱,
    .zip이면 안에 들어있는 첫 번째 .xml 파일을 열어서 파싱한다.
    """
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            xml_names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if not xml_names:
                raise ValueError("ZIP 안에 .xml 파일을 찾지 못했습니다.")
            xml_name = xml_names[0]
            xml_bytes = z.read(xml_name)
        root = ET.fromstring(xml_bytes)
    else:
        tree = ET.parse(path)
        root = tree.getroot()
    return root


def flatten_text(elem):
    """XML element 안의 모든 텍스트를 평탄화해서 하나의 문자열로."""
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(flatten_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def get_text(elem):
    """Element에서 텍스트만 깔끔하게 추출 (공백 정리)."""
    if elem is None:
        return None
    text_parts = [t.strip() for t in elem.itertext() if t and t.strip()]
    return " ".join(text_parts) if text_parts else None


def extract_basic_info_from_spl(root):
    """
    SPL XML에서 NDC, 회사명, 제형 후보를 뽑는다.
    """
    # 1) NDC 후보
    ndc_list = []
    for id_el in root.findall(".//hl7:id", NS):
        ext = (id_el.get("extension") or "").strip()
        if not ext:
            continue
        if any(ch.isdigit() for ch in ext) and "-" in ext and 9 <= len(ext) <= 15:
            if ext not in ndc_list:
                ndc_list.append(ext)
    ndc_text = "\n".join(ndc_list)

    # 2) 회사명
    company_list = []
    for org_name in root.findall(".//hl7:representedOrganization/hl7:name", NS):
        txt = flatten_text(org_name).strip()
        if txt and txt not in company_list:
            company_list.append(txt)
    company_text = "\n".join(company_list)

    # 3) 제형
    dosage_list = []
    for form in root.findall(".//hl7:manufacturedProduct//hl7:formCode", NS):
        disp = (form.get("displayName") or "").strip()
        code = (form.get("code") or "").strip()
        txt = disp or code
        if txt and txt.lower() != "drug form" and txt not in dosage_list:
            dosage_list.append(txt)
    dosage_text = "\n".join(dosage_list)

    return ndc_text, company_text, dosage_text


def find_section_by_display_name(root, display_name: str):
    """code@displayName 이 정확히 display_name 인 section 하나 반환."""
    for sec in root.findall(".//hl7:section", NS):
        code = sec.find("hl7:code", NS)
        if code is not None and code.get("displayName", "").upper() == display_name.upper():
            return sec
    return None


def find_sections_by_display_contains(root, keyword: str):
    """
    code@displayName 안에 keyword(대소문자 무시)가 포함된 모든 section 리스트 반환.
    (예: WARNING, CONTRAINDICATION, PREGNANCY, LACTATION, PEDIATRIC, PHARMACOKINETICS 등)
    """
    key = keyword.upper()
    matched = []
    for sec in root.findall(".//hl7:section", NS):
        code = sec.find("hl7:code", NS)
        if code is not None:
            disp = (code.get("displayName") or "").upper()
            if key in disp:
                matched.append(sec)
    return matched


def find_labor_and_delivery_sections(root):
    """
    displayName 안에 LABOR 와 DELIVERY 가 모두 포함된 section을 찾는다.
    - LABOR & DELIVERY SECTION
    - LABOR AND DELIVERY
    등 다양한 표기를 모두 커버하기 위함.
    """
    matched = []
    for sec in root.findall(".//hl7:section", NS):
        code = sec.find("hl7:code", NS)
        if code is not None:
            disp = (code.get("displayName") or "").upper()
            norm = disp.replace("&", "AND")
            if ("LABOR" in norm) and ("DELIVERY" in norm):
                matched.append(sec)
    return matched


def section_to_plain_text(sec):
    """
    section 안의 <paragraph>들을 줄 단위 텍스트로 변환.
    paragraph가 없으면 <text> 전체를 사용.
    """
    if sec is None:
        return ""

    paras = sec.findall(".//hl7:paragraph", NS)
    texts = []

    if paras:
        for p in paras:
            t = flatten_text(p).strip()
            if t:
                texts.append(t)
    else:
        text_el = sec.find("hl7:text", NS)
        if text_el is not None:
            t = flatten_text(text_el).strip()
            if t:
                texts.append(t)

    return "\n".join(texts)


def sections_to_text_block(sections):
    texts = []
    for sec in sections:
        t = section_to_plain_text(sec)
        if t:
            texts.append(t)
    return "\n".join(texts)


def get_cover_excerpt(root):
    """title + effectiveTime + NDC + 회사명 + 제형까지 COVER 부분 구성."""
    title_el = root.find(".//hl7:title", NS)
    title_text = flatten_text(title_el).strip() if title_el is not None else ""

    eff_el = root.find("./hl7:effectiveTime", NS)
    eff_val = eff_el.get("value") if eff_el is not None else ""

    ndc_text, company_text, dosage_text = extract_basic_info_from_spl(root)

    cover_lines = []
    if title_text:
        cover_lines.append(f"TITLE: {title_text}")
    if eff_val:
        cover_lines.append(f"EFFECTIVE TIME: {eff_val}")
    if ndc_text:
        cover_lines.append(f"NDC: {ndc_text}")
    if company_text:
        cover_lines.append(f"COMPANY: {company_text}")
    if dosage_text:
        cover_lines.append(f"DOSAGE FORM: {dosage_text}")

    return "\n".join(cover_lines)


def extract_drug_and_ingredient_for_prompt(root):
    """
    subject > manufacturedProduct 를 순회하면서
    - drug_name 후보들 (제품명)
    - active_ingredient 후보들 (모든 활성 성분 이름 + activeMoiety 이름)
    을 전부 모아서 세미콜론(;)으로 join해서 반환.
    """
    drug_names = set()
    ingredients = set()

    for mp_container in root.findall('.//hl7:subject/hl7:manufacturedProduct', NS):
        # 새 스타일
        mp_new = mp_container.find('hl7:manufacturedProduct', NS)
        if mp_new is not None:
            drug_name = get_text(mp_new.find('hl7:name', NS))
            if drug_name:
                drug_names.add(drug_name)

            for ing in mp_new.findall('hl7:ingredient', NS):
                cls = (ing.get('classCode') or '')
                if cls.startswith('ACT'):
                    ing_name_elem = ing.find('hl7:ingredientSubstance/hl7:name', NS)
                    ing_name = get_text(ing_name_elem)
                    if ing_name:
                        ingredients.add(ing_name)

                    am_elem = ing.find('.//hl7:activeMoiety/hl7:name', NS)
                    am_name = get_text(am_elem)
                    if am_name:
                        ingredients.add(am_name)

        # 옛 스타일
        mm = mp_container.find('hl7:manufacturedMedicine', NS)
        if mm is not None:
            drug_name = get_text(mm.find('hl7:name', NS))
            if drug_name:
                drug_names.add(drug_name)

            for ai_elem in mm.findall('hl7:activeIngredient', NS):
                ai_name_elem = ai_elem.find('hl7:activeIngredientSubstance/hl7:name', NS)
                ai_name = get_text(ai_name_elem)
                if ai_name:
                    ingredients.add(ai_name)

                am_elem = ai_elem.find('.//hl7:activeMoiety/hl7:name', NS)
                am_name = get_text(am_elem)
                if am_name:
                    ingredients.add(am_name)

    drug_name_str = "; ".join(sorted(drug_names)) if drug_names else ""
    ingredient_str = "; ".join(sorted(ingredients)) if ingredients else ""

    return drug_name_str, ingredient_str


def build_label_text_for_placenta(root):
    """
    태반 통과 판단용으로 다음 섹션들을 모아 label_text를 만든다.
    - CONTRAINDICATIONS (4번 + 'CONTRAINDICATION' 키워드)
    - WARNINGS AND PRECAUTIONS (5번 + 'WARNING' 키워드)
    - PREGNANCY (8.1)
    - LABOR AND DELIVERY / LABOR & DELIVERY SECTION (주로 8.2)
    - LACTATION / NURSING MOTHERS / BREASTFEEDING
    - PEDIATRIC USE / PEDIATRICS
    - CLINICAL PHARMACOLOGY / PHARMACOKINETICS (12.3 + 'PHARMACOKINETICS' 키워드)
    """
    # 1) 숫자 기반 섹션
    sec4  = find_section_by_display_name(root, "CONTRAINDICATIONS SECTION")
    sec5  = find_section_by_display_name(root, "WARNINGS AND PRECAUTIONS SECTION")
    sec8  = find_section_by_display_name(root, "PREGNANCY SECTION")
    sec12 = find_section_by_display_name(root, "CLINICAL PHARMACOLOGY SECTION")

    contra_sections     = []
    warning_sections    = []
    pregnancy_sections  = []
    pk_sections         = []

    if sec4 is not None:
        contra_sections.append(sec4)
    if sec5 is not None:
        warning_sections.append(sec5)
    if sec8 is not None:
        pregnancy_sections.append(sec8)
    if sec12 is not None:
        pk_sections.append(sec12)

    # 2) 키워드 기반 섹션들

    # CONTRAINDICATION
    for sec in find_sections_by_display_contains(root, "CONTRAINDICATION"):
        if sec not in contra_sections:
            contra_sections.append(sec)

    # WARNINGS
    for sec in find_sections_by_display_contains(root, "WARNING"):
        if sec not in warning_sections:
            warning_sections.append(sec)

    # PREGNANCY
    for sec in find_sections_by_display_contains(root, "PREGNANCY"):
        if sec not in pregnancy_sections:
            pregnancy_sections.append(sec)

    # LABOR AND DELIVERY (LABOR & DELIVERY SECTION 포함)
    for sec in find_labor_and_delivery_sections(root):
        if sec not in pregnancy_sections:
            pregnancy_sections.append(sec)

    # LACTATION / NURSING MOTHERS / BREASTFEEDING
    for key in ["LACTATION", "NURSING MOTHERS", "BREASTFEEDING"]:
        for sec in find_sections_by_display_contains(root, key):
            if sec not in pregnancy_sections:
                pregnancy_sections.append(sec)

    # PEDIATRIC USE / PEDIATRICS
    for key in ["PEDIATRIC", "PEDIATRICS"]:
        for sec in find_sections_by_display_contains(root, key):
            if sec not in pregnancy_sections:
                pregnancy_sections.append(sec)

    # PHARMACOKINETICS
    for sec in find_sections_by_display_contains(root, "PHARMACOKINETICS"):
        if sec not in pk_sections:
            pk_sections.append(sec)

    # 3) COVER 요약 + 전체 label_text 구성
    cover_excerpt = get_cover_excerpt(root)

    label_text = (
        "===== COVER (EXCERPT) =====\n"
        + cover_excerpt
        + "\n\n===== CONTRAINDICATIONS (INCLUDING SECTION 4 & KEYWORD MATCH) =====\n"
        + "[[CONTRA_START]]\n"
        + (sections_to_text_block(contra_sections) or "(contraindications 관련 텍스트 없음)")
        + "\n[[CONTRA_END]]\n\n"
        + "===== WARNINGS AND PRECAUTIONS (INCLUDING SECTION 5 & KEYWORD MATCH) =====\n"
        + "[[WARNINGS_START]]\n"
        + (sections_to_text_block(warning_sections) or "(warnings 관련 텍스트 없음)")
        + "\n[[WARNINGS_END]]\n\n"
        + "===== PREGNANCY / LABOR AND DELIVERY / LACTATION / PEDIATRIC USE =====\n"
        + "[[PREGNANCY_START]]\n"
        + (sections_to_text_block(pregnancy_sections) or "(pregnancy / labor and delivery / lactation / pediatric 관련 텍스트 없음)")
        + "\n[[PREGNANCY_END]]\n\n"
        + "===== PHARMACOKINETICS (INCLUDING SECTION 12.3 & KEYWORD MATCH) =====\n"
        + "[[PK_START]]\n"
        + (sections_to_text_block(pk_sections) or "(pharmacokinetics 관련 텍스트 없음)")
        + "\n[[PK_END]]\n"
    )

    return label_text


# ================== 3. 1차 LLM: 태반 통과 '근거만' 추출 ==================
placenta_parser = JsonOutputParser()

PLACENTA_SYSTEM_PROMPT_RAW = """
[역할]
너는 'FDA 임부/태반 정보 분석기'야.
의역/창작 금지. evidence 문장은 원문 그대로 복사해야 한다.

[입력]
- drug_name: 라벨에서 추출한 약물 이름 (여러 개면 ; 로 구분)
- active_ingredient: 주성분 이름(들) (여러 개면 ; 로 구분)
- label_text:
  - [[CONTRA_START]]~[[CONTRA_END]]: CONTRAINDICATIONS 관련 텍스트 (주로 4번)
  - [[WARNINGS_START]]~[[WARNINGS_END]]: WARNINGS AND PRECAUTIONS 관련 텍스트 (주로 5번)
  - [[PREGNANCY_START]]~[[PREGNANCY_END]]: PREGNANCY / LABOR AND DELIVERY / LACTATION / PEDIATRIC USE 관련 텍스트 (주로 8.x, Pediatrics)
  - [[PK_START]]~[[PK_END]]: CLINICAL PHARMACOLOGY/PHARMACOKINETICS 관련 텍스트 (주로 12.3)

[목표]
- 약물이 태반을 통과하는지 여부를 '판단'하지 말고,
- **태반 통과 여부를 판단할 때 근거가 될 수 있는 문장들만** 골라서 정리한다.
- "crosses / no_cross / unknown" 같은 판단 레이블은 절대 만들지 마라.

[태반 통과 관련 키워드 예시]
- placenta, placental, transplacental, placental transfer, placental passage
- cross(es) the placenta, cross(es) the placental barrier
- umbilical cord blood, cord blood, amniotic fluid, fetal plasma, fetal blood, fetal tissue, fetal concentrations, fetal levels 등

[추출 규칙]
1) evidence_sentences
   - 태반을 통과하는지/안 하는지/불명확한지에 대한 직접적인 단서가 되는 문장만 담는다.
   - 예:
     - "Drug X crosses the placenta."
     - "No placental transfer of Drug X was observed in animal studies."
     - "Drug concentrations in umbilical cord blood were similar to maternal plasma."
   - 한 문장 안에 여러 정보가 있어도 통째로 복사한다.
   - evidence_sentences 안에서는 **원문 그대로** 복사해야 한다:
     - 문장부호, 철자, 대소문자, 줄바꿈을 고치지 말고 그대로 사용.
     - 새로운 내용을 만들어내면 안 된다.
     - 여러 문장을 합쳐서 요약하거나 재작성하면 안 된다.
     - 문장 일부만 떼어서 재구성하면 안 된다. 항상 완전한 원문 문장 단위로 사용한다.
   - 같은 내용을 반복하는 문장은 1개만 선택해도 좋다.
   - 태반, 태아 혈액/양수/제대혈 농도 등과 관련된 문장이면 폭넓게 포함해도 된다.

2) 판단 금지
   - 이 JSON 안에서는 "태반을 통과한다/안 한다/불명" 같은 결론을 내리지 말고,
     오직 '판단에 사용할 수 있는 원문 문장 리스트(evidence_sentences)'만 제공한다.

[출력 형식]
반드시 아래 스키마를 만족하는 유효한 JSON 객체만 출력하라. 설명 문장은 붙이지 마라.

{
  "drug_name": "<입력 drug_name 그대로 또는 보정한 값>",
  "active_ingredient": "<입력 active_ingredient 그대로 또는 보정한 값>",
  "evidence_sentences": [
    "<태반 통과 관련 근거 문장 1 (원문 그대로)>",
    "<태반 통과 관련 근거 문장 2 (원문 그대로)>"
  ]
}
"""

PLACENTA_SYSTEM_PROMPT_ESCAPED = PLACENTA_SYSTEM_PROMPT_RAW.replace("{", "{{").replace("}", "}}")

placenta_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PLACENTA_SYSTEM_PROMPT_ESCAPED),
        ("system", "{format_instructions}"),
        (
            "human",
            "아래 정보를 사용해서 태반 통과 여부를 판단하는 데 근거가 될 수 있는 문장들만 골라라.\n"
            "직접적인 판단(crosses / no_cross / unknown)은 하지 말고, 근거 문장만 추출하라.\n\n"
            "drug_name:\n{drug_name}\n\n"
            "active_ingredient:\n{active_ingredient}\n\n"
            "label_text (4/5/8.x/LACTATION/PEDIATRIC/12.3 및 WARNING/CONTRAINDICATION/PREGNANCY/"
            "LABOR AND DELIVERY/LACTATION/PEDIATRIC USE/PHARMACOKINETICS 관련 섹션):\n\n"
            "{label_text}"
        ),
    ]
).partial(format_instructions=placenta_parser.get_format_instructions())

placenta_chain = placenta_prompt | llm | placenta_parser


# ------- 3-1. 1차 evidence가 실제 label_text 안에 있는지 검증하는 필터 --------
def normalize_for_match(text: str) -> str:
    """
    공백/줄바꿈을 정규화해서 비교하기 쉽게 만든다.
    - 여러 칸 공백, 줄바꿈 등을 모두 단일 공백으로 치환
    - 앞뒤 공백 제거
    """
    if text is None:
        return ""
    return " ".join(str(text).split())


def filter_evidence_sentences_by_label_text(result: dict, label_text: str) -> dict:
    """
    1차 LLM 결과(result["evidence_sentences"]) 중에서
    실제로 label_text 안에 존재하는 문장만 남긴다.
    (없는 문장 = LLM이 지어낸 문장 → 제거)
    """
    evidence_list = result.get("evidence_sentences") or []
    if not evidence_list:
        result["evidence_sentences"] = []
        return result

    norm_label = normalize_for_match(label_text)
    filtered = []
    dropped = []

    for sent in evidence_list:
        if not sent:
            continue
        norm_sent = normalize_for_match(sent)
        if norm_sent and norm_sent in norm_label:
            filtered.append(sent)
        else:
            dropped.append(sent)

    result["evidence_sentences"] = filtered
    if dropped:
        # 디버깅용으로 남겨두고 싶으면 이 키를 나중에 참고 가능
        result["_dropped_evidence_not_in_label"] = dropped

    return result


def run_placenta_chain_for_file(xml_path: str):
    """
    한 개 XML 파일에 대해:
    - drug_name / active_ingredient 추출
    - label_text 구성
    - 1차 LLM 호출 → evidence_sentences
    - evidence_sentences 중 실제 label_text 안에 존재하는 문장만 필터링
    """
    root = load_spl_root(xml_path)
    drug_name, active_ing = extract_drug_and_ingredient_for_prompt(root)
    label_text = build_label_text_for_placenta(root)

    result = call_with_backoff(
        placenta_chain.invoke,
        {
            "drug_name": drug_name,
            "active_ingredient": active_ing,
            "label_text": label_text,
        },
    )

    result = filter_evidence_sentences_by_label_text(result, label_text)

    return result


# ================== 4. 1차 결과 CSV 저장 유틸 ==================
def append_placenta_result_to_csv(xml_path: str, result: dict, csv_path: str):
    """
    LLM 1차 결과(JSON)를 CSV 한 행으로 추가.
    evidence_sentences 리스트는 줄바꿈으로 합쳐서 넣는다.
    """
    folder = os.path.basename(os.path.dirname(xml_path))
    file_name = os.path.basename(xml_path)

    drug_name = result.get("drug_name", "")
    active_ingredient = result.get("active_ingredient", "")
    evidence_list = result.get("evidence_sentences", []) or []

    evidence_text = "\n".join(evidence_list)

    fieldnames = [
        "folder",
        "file",
        "drug_name",
        "active_ingredient",
        "evidence_sentences",
        "raw_json",
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "folder": folder,
                "file": file_name,
                "drug_name": drug_name,
                "active_ingredient": active_ingredient,
                "evidence_sentences": evidence_text,
                "raw_json": json.dumps(result, ensure_ascii=False),
            }
        )


# ================== 5. 여러 XML 파일 수집 & 1차 루프 실행 ==================
def collect_xml_files(root_dir: str):
    xml_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".xml"):
                xml_files.append(os.path.join(dirpath, name))
    xml_files.sort()
    return xml_files


def load_processed_files(log_path: str):
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    processed.add(p)
    return processed


def run_first_stage():
    """
    1차 LLM:
    - selected_xml 아래 모든 XML을 훑어서
    - 태반 관련 evidence_sentences만 추출하고
    - placenta_llm_1st_results.csv에 누적 저장
    - 이번 실행에서 새로 처리하는 파일 수는 MAX_FIRST_STAGE_NEW_FILES로 제한
    """
    xml_files = collect_xml_files(ROOT_DIR)
    print(f"[1차] 총 XML 파일 수: {len(xml_files)}")

    processed_set = load_processed_files(PROCESSED_LOG_PATH)
    print(f"[1차] 이미 처리된 파일 수: {len(processed_set)}")

    new_count = 0

    try:
        for idx, full_path in enumerate(xml_files, start=1):
            if full_path in processed_set:
                print(f"\n[1차][SKIP] 이미 처리된 파일이라 건너뜀 ({idx}/{len(xml_files)}): {full_path}")
                continue

            if MAX_FIRST_STAGE_NEW_FILES and new_count >= MAX_FIRST_STAGE_NEW_FILES:
                print(f"\n[1차] 이번 실행에서 새로 처리할 최대 파일 수({MAX_FIRST_STAGE_NEW_FILES})에 도달하여 1차를 중단합니다.")
                break

            print("\n============================================")
            print(f"[1차][{idx}/{len(xml_files)}] 입력 파일: {full_path}")

            try:
                result = run_placenta_chain_for_file(full_path)

                append_placenta_result_to_csv(full_path, result, PLACENTA_OUTPUT_CSV)

                with open(PROCESSED_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(full_path + "\n")
                processed_set.add(full_path)

                new_count += 1
                print(f"→ [1차] LLM 태반 근거 결과 저장 완료 (이번 실행 새로 처리: {new_count}개)")

            except DailyLimitReached:
                print("\n=== [1차] 일일/분당 토큰 한도 또는 rate limit에 도달하여 실행을 중단했습니다. ===")
                print("나중에 한도가 리셋된 뒤 다시 실행하면,")
                print("placenta_processed_files_log.txt 기준으로 이미 처리된 파일은 건너뛰고 이어서 계속 분석합니다.")
                break

            except Exception as e:
                msg = repr(e)
                print(f"⚠️ [1차] 에러 발생, 이 파일 처리 실패: {full_path}")
                print("에러 내용:", msg)
                with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(full_path + "\t" + msg + "\n")

            if PER_FILE_SLEEP_SECONDS > 0:
                print(f"→ 다음 파일까지 {PER_FILE_SLEEP_SECONDS}초 대기")
                time.sleep(PER_FILE_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n⏹ [1차] 수동으로 중단되었습니다.")


# ================== 6. 2차 LLM: 태반 통과 여부 judgement ==================
judge_parser = JsonOutputParser()

JUDGE_SYSTEM_PROMPT_RAW = """
[역할]
너는 'FDA 임부/태반 정보 평가기'야.

[입력]
- drug_name: 약물 이름
- active_ingredient: 주성분 이름(들)
- evidence_sentences: 1차 LLM이 이미 뽑아둔 '태반 통과와 관련 있을 수 있는' 근거 문장 리스트

[매우 중요한 규칙]
1) **오직 evidence_sentences 안에 있는 문장만** 근거로 사용해서 판단해야 한다.
   - label_text나 외부 상식, 다른 약동학 지식은 사용하지 마라.
   - evidence_sentences에 태반 통과 여부가 직접적으로 나타나지 않으면, 반드시 "unknown"으로 판단해야 한다.

2) placental_transfer_judgement 값 정의
   - "crosses":
     - 예: "crosses the placenta", "placental transfer occurs",
           "drug was detected in cord blood / amniotic fluid / fetal plasma" 등
   - "no_cross":
     - 예: "does not cross the placenta", "no placental transfer was observed" 등
   - "unknown":
     - 위와 같은 표현이 전혀 없거나,
       evidence 문장들만으로는 '통과한다/안 한다'고 말하기 애매한 경우.

3) reason
   - 한국어로 3~4문장 정도로,
     - 어떤 evidence 문장(들)을 근거로 "crosses/no_cross/unknown"이라고 판단했는지 설명한다.

4) evidence_for_judgement
   - judgement를 내리는 데 핵심이 된 evidence 문장만 골라서 넣는다.
   - 없는 경우에는 빈 배열 [] 로 둔다.

5) evidence_sentences가 비어 있으면
   - placental_transfer_judgement는 무조건 "unknown"으로 설정한다.
   - reason에는 "근거 문장이 없어 태반 통과 여부를 판단할 수 없다"는 취지를 한국어로 적어라.

[출력 형식]
반드시 아래 스키마를 만족하는 유효한 JSON 객체만 출력하라. 설명 문장은 붙이지 마라.

{
  "placental_transfer_judgement": "crosses | no_cross | unknown",
  "reason": "<한국어 설명>",
  "evidence_for_judgement": [
    "<judgement의 근거가 된 evidence 문장 1>",
    "<judgement의 근거가 된 evidence 문장 2>"
  ]
}
"""

JUDGE_SYSTEM_PROMPT_ESCAPED = JUDGE_SYSTEM_PROMPT_RAW.replace("{", "{{").replace("}", "}}")

judge_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT_ESCAPED),
        ("system", "{format_instructions}"),
        (
            "human",
            "아래 정보를 바탕으로 약물이 태반을 통과하는지에 대한 판단을 내려라.\n"
            "- 단, evidence_sentences 안에 있는 문장만 근거로 사용해야 한다.\n"
            "- 외부 상식이나 다른 자료는 절대 사용하지 마라.\n\n"
            "drug_name:\n{drug_name}\n\n"
            "active_ingredient:\n{active_ingredient}\n\n"
            "evidence_sentences (각 항목은 하나의 근거 문장):\n\n"
            "{evidence_block}"
        ),
    ]
).partial(format_instructions=judge_parser.get_format_instructions())

judge_chain = judge_prompt | llm | judge_parser


def run_judge_llm(drug_name: str, active_ingredient: str, evidence_list):
    """
    2차 LLM 호출:
    - evidence_list: 문자열 리스트 (각각 하나의 근거 문장)
    """
    if evidence_list:
        lines = []
        for i, sent in enumerate(evidence_list, start=1):
            lines.append(f"{i}. {sent}")
        evidence_block = "\n".join(lines)
    else:
        evidence_block = "(evidence_sentences가 비어 있음)"

    resp = call_with_backoff(
        judge_chain.invoke,
        {
            "drug_name": drug_name or "",
            "active_ingredient": active_ingredient or "",
            "evidence_block": evidence_block,
        },
    )
    return resp


# ================== 7. 2차 결과 CSV 및 로그 유틸 ==================
def load_processed_keys(log_path: str):
    """
    2차 판단이 이미 끝난 row들을 'folder|file' 키로 저장해두고,
    재실행 시 건너뛰기 위함.
    """
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key:
                    processed.add(key)
    return processed


def append_processed_key(log_path: str, key: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(key + "\n")


def append_judgement_to_csv(row, judge_result: dict, output_csv_path: str):
    """
    row: 1차 CSV의 한 행(dict)
    judge_result: 2차 LLM JSON 결과
    output_csv_path: 2차 결과 CSV 경로
    """
    folder = row.get("folder", "")
    file_name = row.get("file", "")

    drug_name_1 = row.get("drug_name", "")
    active_ing_1 = row.get("active_ingredient", "")
    all_evidence_text = row.get("evidence_sentences", "")

    judgement = judge_result.get("placental_transfer_judgement", "")
    reason = judge_result.get("reason", "")
    evidence_for_judgement_list = judge_result.get("evidence_for_judgement", []) or []
    evidence_for_judgement_text = "\n".join(evidence_for_judgement_list)

    fieldnames = [
        "folder",
        "file",
        "drug_name_first",
        "active_ingredient_first",
        "placental_transfer_judgement",
        "reason",
        "evidence_for_judgement",
        "all_evidence_sentences",
    ]

    file_exists = os.path.exists(output_csv_path)

    with open(output_csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "folder": folder,
                "file": file_name,
                "drug_name_first": drug_name_1,
                "active_ingredient_first": active_ing_1,
                "placental_transfer_judgement": judgement,
                "reason": reason,
                "evidence_for_judgement": evidence_for_judgement_text,
                "all_evidence_sentences": all_evidence_text,
            }
        )


def run_second_stage():
    """
    2차 LLM:
    - 1차 CSV를 읽어서 evidence_sentences만을 근거로 태반 통과 여부를 판단
    - 2차 CSV에 저장
    """
    if not os.path.exists(FIRST_STAGE_CSV):
        print(f"[2차][ERROR] 1차 결과 CSV를 찾을 수 없습니다: {FIRST_STAGE_CSV}")
        return

    processed_keys = load_processed_keys(JUDGE_PROCESSED_LOG)
    print(f"[2차] 이미 판단 완료된 row 개수: {len(processed_keys)}")

    with open(FIRST_STAGE_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"[2차] 1차 CSV 전체 row 수: {total}")

    new_count = 0

    try:
        for idx, row in enumerate(rows, start=1):
            folder = row.get("folder", "")
            file_name = row.get("file", "")
            key = f"{folder}|{file_name}"

            if key in processed_keys:
                print(f"[2차][SKIP] 이미 판단 완료 ({idx}/{total}): {key}")
                continue

            if MAX_SECOND_STAGE_NEW_ROWS and new_count >= MAX_SECOND_STAGE_NEW_ROWS:
                print(f"\n[2차] 이번 실행에서 새로 판단할 최대 row 수({MAX_SECOND_STAGE_NEW_ROWS})에 도달하여 2차를 중단합니다.")
                break

            print("\n============================================")
            print(f"[2차][{idx}/{total}] 판단 대상: folder={folder}, file={file_name}")

            try:
                evidence_list = []
                raw_json_1 = row.get("raw_json", "")
                if raw_json_1:
                    try:
                        data_1 = json.loads(raw_json_1)
                        evidence_list = data_1.get("evidence_sentences", []) or []
                    except Exception:
                        evidence_list = []

                if (not evidence_list) and row.get("evidence_sentences"):
                    evidence_list = [
                        s for s in row["evidence_sentences"].splitlines() if s.strip()
                    ]

                drug_name = row.get("drug_name", "") or ""
                active_ingredient = row.get("active_ingredient", "") or ""

                judge_result = run_judge_llm(drug_name, active_ingredient, evidence_list)

                append_judgement_to_csv(row, judge_result, SECOND_STAGE_CSV)

                append_processed_key(JUDGE_PROCESSED_LOG, key)

                new_count += 1
                print(
                    f"→ [2차] 판단 완료: judgement={judge_result.get('placental_transfer_judgement', '')} "
                    f"(이번 실행 새로 판단: {new_count}개)"
                )

            except DailyLimitReached:
                print("\n=== [2차] 일일/분당 토큰 한도 또는 rate limit에 도달하여 실행을 중단했습니다. ===")
                print("나중에 한도가 리셋된 뒤 이 스크립트를 다시 실행하면,")
                print("placenta_judgement_processed_log.txt 기준으로 이미 처리된 row는 건너뛰고 이어서 계속 분석합니다.")
                break

            except Exception as e:
                msg = repr(e)
                print(f"⚠️ [2차] 에러 발생, 이 row 처리 실패: {key}")
                print("에러 내용:", msg)
                with open(JUDGE_ERROR_LOG, "a", encoding="utf-8") as ef:
                    ef.write(key + "\t" + msg + "\n")

            if PER_ROW_SLEEP_SECONDS > 0:
                print(f"→ 다음 row까지 {PER_ROW_SLEEP_SECONDS}초 대기")
                time.sleep(PER_ROW_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n⏹ [2차] 수동으로 중단되었습니다.")




# ================== 8. 최종 non-unknown 결과 생성 ==================
def finalize_non_unknown_results():
    """
    Stage 2 결과에서 'unknown'을 제외하고
    crosses / no_cross 결과만 별도 CSV로 저장한다.

    Historical manuscript run:
    - Input curated XML labels: 2,836
    - Retained non-unknown FDA labels: 610

    The current API result may not be bit-for-bit identical in the future,
    even at temperature=0, because hosted LLM services/models can change.
    """

    if not os.path.exists(SECOND_STAGE_CSV):
        print(
            "[FINAL] Stage 2 CSV가 아직 없어 "
            "최종 결과를 생성하지 않습니다."
        )
        return None

    with open(
        SECOND_STAGE_CSV,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    counts = Counter()

    retained_rows = []

    for row in rows:
        judgement = (
            row.get(
                "placental_transfer_judgement",
                "",
            )
            or ""
        ).strip().lower()

        counts[
            judgement
            if judgement
            else "(blank)"
        ] += 1

        if judgement in {
            "crosses",
            "no_cross",
        }:
            retained_rows.append(
                row
            )

    with open(
        RETAINED_OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            retained_rows
        )

    xml_count = len(
        collect_xml_files(
            ROOT_DIR
        )
    )

    processed_stage1 = len(
        load_processed_files(
            PROCESSED_LOG_PATH
        )
    )

    processed_stage2 = len(
        load_processed_keys(
            JUDGE_PROCESSED_LOG
        )
    )

    summary = {
        "selected_xml_count": xml_count,
        "expected_selected_xml_count": EXPECTED_XML_COUNT,
        "stage1_processed_log_count": processed_stage1,
        "stage1_csv_row_count": (
            sum(
                1
                for _ in open(
                    PLACENTA_OUTPUT_CSV,
                    "r",
                    encoding="utf-8-sig",
                )
            )
            - 1
            if os.path.exists(
                PLACENTA_OUTPUT_CSV
            )
            else 0
        ),
        "stage2_processed_log_count": processed_stage2,
        "stage2_csv_row_count": len(
            rows
        ),
        "judgement_counts": dict(
            counts
        ),
        "retained_non_unknown_count": len(
            retained_rows
        ),
        "historical_retained_count": HISTORICAL_RETAINED_COUNT,
        "retained_output_csv": str(
            Path(
                RETAINED_OUTPUT_CSV
            ).relative_to(
                PROJECT_ROOT
            )
        ),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "LLM external-labeling reproduction summary"
    )
    print(
        "=" * 80
    )

    print(
        f"Selected XML files: {xml_count}"
    )
    print(
        f"Stage 1 processed: {processed_stage1}"
    )
    print(
        f"Stage 2 processed: {processed_stage2}"
    )
    print(
        f"Stage 2 CSV rows: {len(rows)}"
    )

    print(
        "\nJudgement counts:"
    )

    for label, count in sorted(
        counts.items()
    ):
        print(
            f"  {label}: {count}"
        )

    print(
        f"\nRetained (crosses + no_cross): "
        f"{len(retained_rows)}"
    )

    if (
        xml_count
        == EXPECTED_XML_COUNT
        and len(rows)
        == EXPECTED_XML_COUNT
    ):
        if (
            len(
                retained_rows
            )
            == HISTORICAL_RETAINED_COUNT
        ):
            print(
                "Historical retained count reproduced: 610"
            )
        else:
            print(
                "WARNING: Full 2,836-label run completed, "
                f"but retained count was {len(retained_rows)} "
                f"instead of the historical {HISTORICAL_RETAINED_COUNT}. "
                "Hosted LLM outputs may change over time even at temperature=0."
            )
    else:
        print(
            "NOTE: The full 2,836-label run is not yet complete. "
            "Rerun the notebook to continue from the processed logs."
        )

    print(
        f"\n[RETAINED OUTPUT]\n"
        f"{RETAINED_OUTPUT_CSV}"
    )

    return summary


def save_reproduction_settings():
    """
    Save non-secret reproducibility settings.
    """

    settings = {
        "analysis": (
            "Two-stage LLM extraction and classification "
            "of curated FDA label XML files"
        ),
        "input_directory": str(
            Path(
                ROOT_DIR
            ).relative_to(
                PROJECT_ROOT
            )
        ),
        "expected_xml_count": EXPECTED_XML_COUNT,
        "historical_retained_non_unknown_count": HISTORICAL_RETAINED_COUNT,
        "provider": "Groq API",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0,
        "max_tokens": 4096,
        "stage1_task": (
            "Extract verbatim evidence sentences relevant to placental transfer"
        ),
        "stage1_evidence_validation": (
            "Keep only extracted evidence that is present in the source label_text "
            "after whitespace normalization"
        ),
        "stage2_task": (
            "Classify each label as crosses, no_cross, or unknown "
            "using only Stage 1 evidence sentences"
        ),
        "final_filter": (
            "Retain crosses and no_cross; exclude unknown"
        ),
        "batch_limits": {
            "stage1": MAX_FIRST_STAGE_NEW_FILES,
            "stage2": MAX_SECOND_STAGE_NEW_ROWS,
        },
        "api_key_storage": (
            "Not saved. Read from GROQ_API_KEY environment variable "
            "or entered interactively via getpass."
        ),
    }

    with open(
        SETTINGS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            settings,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ================== 9. 전체 파이프라인 실행 ==================
if __name__ == "__main__":

    xml_files = collect_xml_files(
        ROOT_DIR
    )

    print(
        "=" * 80
    )
    print(
        "FDA label LLM reproduction pipeline"
    )
    print(
        "=" * 80
    )

    print(
        f"Input directory: {ROOT_DIR}"
    )
    print(
        f"XML file count: {len(xml_files)}"
    )

    if (
        len(xml_files)
        != EXPECTED_XML_COUNT
    ):
        print(
            "WARNING: Expected 2,836 curated XML files, "
            f"but found {len(xml_files)}."
        )

    save_reproduction_settings()

    print(
        "\n===== Stage 1: evidence sentence extraction ====="
    )
    run_first_stage()

    print(
        "\n===== Stage 2: placental-transfer judgement ====="
    )
    run_second_stage()

    print(
        "\n===== Final: exclude unknown labels ====="
    )
    finalize_non_unknown_results()

    print(
        "\nPipeline run finished. "
        "If a rate limit stopped processing, rerun this notebook; "
        "the log files will skip already completed items."
    )
