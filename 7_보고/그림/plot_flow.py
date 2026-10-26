"""16:9 flowchart of the tool-based analysis pipeline (for slides). Starts from preprocessed reviews;
raw-data cleanup (HTML entities etc.) is drawn separately in plot_preprocess.py."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

OUT = Path(__file__).resolve().parent  # images sit next to the scripts in 7_보고/그림
plt.rcParams['font.family'] = 'Malgun Gothic'
INK, MUTED = '#14233d', '#5b6a82'
KIND = {  # fill, edge
    'model': ('#e3edfb', '#2a78d6'),   # trained by us
    'tool': ('#eef1f5', '#7d8899'),    # used as is
    'data': ('#fdf1dc', '#d08a14'),
    'result': ('#e4f4ea', '#1d8a4c'),
}


def box(ax, x, y, w, h, kind, title, lines, example=None, tag=None):
    fill, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h - 0.028, title, ha='center', va='top', fontsize=12, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h - 0.075, '\n'.join(lines), ha='center', va='top', fontsize=9, color=INK, linespacing=1.45)
    if example:
        ax.text(x + w / 2, y + 0.018, example, ha='center', va='bottom', fontsize=8.6, color=MUTED, style='italic', linespacing=1.35)
    if tag:
        ax.text(x + w - 0.006, y + h + 0.006, tag, ha='right', va='bottom', fontsize=8, color=edge, fontweight='bold')


def arrow(ax, a, b, dashed=False, color=INK, text=None, text_xy=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, lw=1.4, color=color,
                                 linestyle=(0, (4, 3)) if dashed else '-', shrinkA=2, shrinkB=2))
    if text:
        tx, ty = text_xy or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ax.text(tx, ty, text, ha='center', va='center', fontsize=8.5, color=color, bbox=dict(fc='white', ec='none', pad=1.2))


def main():
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.03, 0.95, '비비수산 리뷰 분석 흐름', fontsize=22, fontweight='bold', color=INK, va='top')
    ax.text(0.03, 0.895, '정리된 리뷰를 의견 단위로 나누고, 긍정·부정을 판단한 뒤, 같은 이유끼리 묶어 상품별로 집계한다',
            fontsize=12, color=MUTED, va='top')
    ax.text(0.03, 0.85, '각 상자 아래 기울임 글씨: 예시 리뷰 한 건이 단계마다 바뀌는 모습', fontsize=9.5, color=MUTED, va='top', style='italic')

    y, h, w, gap, x0 = 0.53, 0.25, 0.137, 0.025, 0.03
    xs = [x0 + i * (w + gap) for i in range(6)]
    steps = [
        ('data', '정리된 리뷰', ['전처리 단계의 결과', '64만 건 · 상품 ID별', '(HTML 기호 · 공백 정리 완료)'], '"치즈 고소한데…\n잘 안 늘어나요ㅠ"', None),
        ('tool', '① 의견 단위 나누기', ['형태소 분석기 Kiwi', '문장 + "-는데 / -지만', '/ -고 / -면서" 뒤에서'], '[치즈 고소한데]\n[잘 안 늘어나요ㅠ]', '학습 없음'),
        ('model', '② 감성 분류', ['식품 학습 KcELECTRA', '긍정 / 부정 / 평가 아님', '+ 확신도'], '긍정 0.97\n부정 0.95', '직접 학습'),
        ('tool', '③ 의미 벡터', ['KURE-v1', '같은 뜻이면', '가까운 좌표'], '1024차원 벡터', '학습 없음'),
        ('tool', '④ 이유별 묶기', ['BERTopic', 'UMAP → HDBSCAN', '핵심 단어 · 기타 분리'], '"치즈가 안 늘어남"\n묶음에 배정', '학습 없음'),
        ('result', '상품별 결과', ['왜 긍정 · 왜 부정인지', '묶음별 건수', '최근 3개월 변화'], '부정 이유 상위 ·\n100건당 최근/이전', None),
    ]
    for x, (kind, title, lines, ex, tag) in zip(xs, steps):
        box(ax, x, y, w, h, kind, title, lines, ex, tag)
    for i in range(5):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))

    # support row: training and validation inputs
    sy, sh, sw = 0.15, 0.18, 0.15
    c2 = xs[2] + w / 2
    sx = {'stage1': xs[1] - 0.02, 'gold': xs[2] + 0.005, 'queue': xs[3] + 0.04, 'pairs': xs[4] + 0.075}
    box(ax, sx['stage1'], sy, sw, sh, 'model', 'Stage 1 사후학습',
        ['KcELECTRA를 비비수산', '리뷰 5만 건으로 추가 학습', '(구어체 · 식품 표현)'], tag='직접 학습')
    box(ax, sx['gold'], sy, sw, sh, 'data', '정답 데이터',
        ['리뷰 300건 → 단위 739개', 'GPT 초안 → Claude 검토', '→ 사람 확정'])
    box(ax, sx['queue'], sy, sw, sh, 'data', '재검수 목록',
        ['확신도 0.8 미만 단위', '사람이 긍정 / 부정 /', '평가 아님 확인 (예정)'])
    box(ax, sx['pairs'], sy, sw, sh, 'data', '같은 뜻 판정',
        ['의견 구절 148짝', '사람이 O / X 판정', 'KURE 선택 근거 (0.92)'])

    arrow(ax, (sx['stage1'] + sw / 2, sy + sh), (c2 - 0.035, y), dashed=True, color='#2a78d6', text='백본으로 사용', text_xy=(xs[1] + 0.06, 0.43))
    arrow(ax, (sx['gold'] + sw / 2, sy + sh), (c2, y), dashed=True, color='#d08a14', text='미세조정', text_xy=(c2 + 0.014, 0.44))
    arrow(ax, (c2 + 0.045, y), (sx['queue'] + 0.05, sy + sh), dashed=True, color='#d08a14', text='확신도 낮으면', text_xy=(xs[3] + 0.03, 0.45))
    arrow(ax, (sx['queue'], sy + 0.06), (sx['gold'] + sw, sy + 0.06), dashed=True, color='#d08a14', text='확인 후 추가', text_xy=((sx['queue'] + sx['gold'] + sw) / 2, sy + 0.025))
    arrow(ax, (sx['pairs'] + sw / 2, sy + sh), (xs[3] + w / 2 + 0.02, y), dashed=True, color='#d08a14', text='모델 선택', text_xy=(xs[4] + 0.06, 0.44))

    lx, ly = 0.03, 0.035
    for i, (kind, label) in enumerate([('model', '직접 학습하는 모델'), ('tool', '학습 없이 쓰는 도구'), ('data', '데이터'), ('result', '결과')]):
        fill, edge = KIND[kind]
        ax.add_patch(FancyBboxPatch((lx + i * 0.16, ly), 0.022, 0.026, boxstyle='round,pad=0.002,rounding_size=0.004', fc=fill, ec=edge, lw=1.4))
        ax.text(lx + i * 0.16 + 0.03, ly + 0.013, label, va='center', fontsize=10, color=INK)
    ax.plot([0.69, 0.73], [ly + 0.013] * 2, color=INK, lw=1.4)
    ax.text(0.735, ly + 0.013, '실행 흐름', va='center', fontsize=10, color=INK)
    ax.plot([0.81, 0.85], [ly + 0.013] * 2, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    ax.text(0.855, ly + 0.013, '학습 · 검증', va='center', fontsize=10, color=INK)
    path = OUT / 'pipeline-flow.png'
    fig.savefig(path, facecolor='white')
    print(path)


if __name__ == '__main__':
    main()
