"""16:9 flowchart of raw-data preprocessing (01 원본 -> 02 상품정리 -> 03 증분 반영), for slides.

Numbers come from 1_전처리/결과/processed/canonical_products/manifest.json;
column rules from src/bibi_reviews/backbone.py (ReviewStore.prepare, normalize_text).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

from plot_flow import KIND, INK, MUTED  # noqa: E402

OUT = Path(__file__).resolve().parent  # images sit next to the scripts in 7_보고/그림
plt.rcParams['font.family'] = 'Malgun Gothic'


def box(ax, x, y, w, h, kind, title, lines, example=None, note=None):
    fill, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012', fc=fill, ec=edge, lw=1.6))
    ax.text(x + 0.012, y + h - 0.022, title, ha='left', va='top', fontsize=12, fontweight='bold', color=INK)
    ax.text(x + 0.012, y + h - 0.068, '\n'.join(lines), ha='left', va='top', fontsize=8.9, color=INK, linespacing=1.5)
    if example:
        ax.text(x + 0.012, y + 0.016, example, ha='left', va='bottom', fontsize=8.4, color=MUTED, style='italic')
    if note:
        ax.text(x + 0.012, y + 0.016, note, ha='left', va='bottom', fontsize=8.4, color='#b06a00', fontweight='bold')


def arrow(ax, a, b, rad=0.0, dashed=False, color=INK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, lw=1.5, color=color,
                                 linestyle=(0, (4, 3)) if dashed else '-', connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))


def main():
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.03, 0.95, '리뷰 원본 데이터 전처리 흐름', fontsize=22, fontweight='bold', color=INK, va='top')
    ax.text(0.03, 0.895, '스마트스토어 원본 CSV를 분석용 데이터로 만드는 과정. 원본 파일은 수정하지 않는다',
            fontsize=12, color=MUTED, va='top')

    w, h, gap, x0 = 0.215, 0.255, 0.02, 0.03
    xs = [x0 + i * (w + gap) for i in range(4)]
    y1, y2 = 0.565, 0.195
    row1 = [
        ('data', '원본 CSV', ['스마트스토어에서 내려받은 그대로', '649,524행 · 25개 열 · 539개 상품', '기간 2025-01-01 ~ 2026-10-13',
                            '(2025-05-31 하루 데이터 없음)'], '리뷰상세내용: "맛은 &hellip;."', None),
        ('tool', '① 필요한 열만 선택 (25 → 10)', ['쓰는 열: 상품번호 · 상품명 · 리뷰글번호 ·', '리뷰상세내용 · 구매자평점 · 리뷰등록일 ·',
                                              '최종수정일 · 리뷰구분 · 관련리뷰글번호 · 전시상태', '뺀 열: 등록자 · 상품주문번호 (개인정보),',
                                              '포토/영상 · 도움수 · 답글 · 베스트 · 이벤트 등'], None, None),
        ('tool', '② 값 검사', ['필수 열 · 필수 값이 비어 있지 않은지', '리뷰글번호 중복이 없는지', '별점이 1~5 정수인지',
                             '전시상태가 정상 / 블라인드인지', '날짜를 한국 시간(KST)으로 통일'], None, '하나라도 어긋나면 파일 전체를 받지 않음'),
        ('tool', '③ 블라인드 제외', ['전시상태 = 정상인 리뷰만 사용', '블라인드 7,352행 제외', '→ 정상 리뷰 642,172행'], None, None),
    ]
    row2 = [
        ('tool', '④ 리뷰 텍스트 정리', ['HTML 기호를 원래 문자로 (&hellip; → …, &lt; → <)', '유니코드 정규화(NFC) · 제어문자 제거',
                                     '연속 공백 · 줄바꿈을 한 칸으로', '원본은 review_text_raw로 따로 보존',
                                     '정리한 문장의 지문(text_hash)으로 같은 문장 찾기'], 'review_text: "맛은 …."', None),
        ('tool', '⑤ 상품명 통일', ['상품 ID마다 대표 이름 1개로 고정', '(리뷰가 가장 많은 이름, 같으면 가나다순)',
                                '이름이 여러 개였던 ID 246개 · 199,045행 변경', '바꾸기 전 이름은 따로 보존'],
         None, '이름이 같은 다른 ID 4쌍은 합치지 않음'),
        ('tool', '⑥ 원래 · 후속 리뷰 묶기', ['관련리뷰글번호로 원래 리뷰와', '한달사용 후기를 한 묶음(review_group_id)',
                                        '학습 · 평가 데이터를 나눌 때', '같은 묶음 · 같은 문장은 같은 쪽으로'], None, None),
        ('result', '분석용 데이터', ['reviews_normal_canonical.parquet', '642,172행 · 539개 상품', '분석 단위 = 상품 ID 하나',
                                 '분석 · 학습은 review_text(정리본) 사용'], None, None),
    ]
    for x, (kind, title, lines, ex, note) in zip(xs, row1):
        box(ax, x, y1, w, h, kind, title, lines, ex, note)
    for x, (kind, title, lines, ex, note) in zip(xs, row2):
        box(ax, x, y2, w, h, kind, title, lines, ex, note)
    for i in range(3):
        arrow(ax, (xs[i] + w, y1 + h / 2), (xs[i + 1], y1 + h / 2))
        arrow(ax, (xs[i] + w, y2 + h / 2), (xs[i + 1], y2 + h / 2))
    mid = (y1 + y2 + h) / 2  # route the row change through the empty band between rows
    ax.plot([xs[3] + w / 2, xs[3] + w / 2, xs[0] + w / 2], [y1 - 0.004, mid, mid], color=INK, lw=1.5)
    arrow(ax, (xs[0] + w / 2, mid), (xs[0] + w / 2, y2 + h))

    ax.text(0.03, 0.105, '새 CSV가 들어오면 (3단계): 같은 ①~⑥을 적용한 뒤 기존 DB와 비교해 신규 · 수정된 리뷰만 반영하고, '
            '바뀌지 않은 리뷰는 건너뛴다 (행마다 지문 row_hash로 비교)', fontsize=10.5, color=INK, va='top')
    lx, ly = 0.03, 0.035
    for i, (kind, label) in enumerate([('tool', '처리 단계 (규칙 · 학습 없음)'), ('data', '원본 데이터'), ('result', '결과')]):
        fill, edge = KIND[kind]
        ax.add_patch(FancyBboxPatch((lx + i * 0.22, ly), 0.022, 0.026, boxstyle='round,pad=0.002,rounding_size=0.004', fc=fill, ec=edge, lw=1.4))
        ax.text(lx + i * 0.22 + 0.03, ly + 0.013, label, va='center', fontsize=10, color=INK)
    ax.text(0.97, ly + 0.013, '코드: src/bibi_reviews/backbone.py · canonicalize_products.py', ha='right', va='center', fontsize=9, color=MUTED)
    path = OUT / 'preprocess-flow.png'
    fig.savefig(path, facecolor='white')
    print(path)


if __name__ == '__main__':
    main()
