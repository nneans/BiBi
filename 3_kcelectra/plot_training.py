"""Learning curve of the post-training (results.json, history.json) -> training-curve.png, for slides."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'
FONT = FontProperties(fname=FONT_PATH)


def main():
    result = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    history = json.loads((OUT / 'history.json').read_text(encoding='utf-8'))
    plt.rcParams['font.family'] = FONT.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    stages = [result['baseline_validation']] + [h['validation'] for h in history]
    fig = plt.figure(figsize=(10, 8.6), dpi=120, facecolor='white')
    text = []
    def label(x, y, value, size=12, color='#203044'):
        text.append(fig.text(x, y, value, fontproperties=FONT, fontsize=size, color=color, va='top'))
    label(.055, .96, 'KcELECTRA 사후학습 · 식품 리뷰', 23)
    label(.055, .906, f'학습 리뷰 {result["training_reviews"]:,}건 · {result["training_products"]}개 상품 · 3 epoch', 13)
    label(.055, .87, f'검증 {result["validation_reviews"]:,}건 · 시험 {result["test_reviews"]:,}건 / 리뷰 전체를 자르지 않고 입력', 11, '#68778b')
    axes = [fig.add_axes([.09, .43, .37, .33]), fig.add_axes([.59, .43, .35, .33])]
    axes[0].plot(range(4), [s['fixed_corruption_token_bce'] for s in stages], marker='o', color='#217d9c', linewidth=2.4)
    axes[0].set_title('검증: 토큰 판별 손실', fontproperties=FONT, fontsize=13, pad=14)
    axes[0].set_ylabel('낮을수록 좋음', fontproperties=FONT)
    axes[1].plot(range(4), [s['replaced_f1'] for s in stages], marker='o', color='#26765d', linewidth=2.4)
    axes[1].set_title('검증: 교체 토큰 검출 F1', fontproperties=FONT, fontsize=13, pad=14)
    axes[1].set_ylabel('높을수록 좋음', fontproperties=FONT)
    for ax in axes:
        ax.set_xticks(range(4), ['학습 전', '1 epoch', '2 epoch', '3 epoch'], fontproperties=FONT, fontsize=9)
        ax.grid(axis='y', color='#e4eaf0')
        ax.spines[['top', 'right']].set_visible(False)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(FONT)
    label(.055, .35, '처음 보는 최종 평가 원문에서도 확인', 16)
    baseline, selected = result['baseline_test'], result['selected_test']
    label(.075, .291, '토큰 판별 손실', 12)
    label(.44, .291, f'{baseline["fixed_corruption_token_bce"]:.3f}  →  {selected["fixed_corruption_token_bce"]:.3f}', 17, '#217d9c')
    label(.075, .235, '교체 토큰 검출 F1', 12)
    label(.44, .235, f'{baseline["replaced_f1"]:.3f}  →  {selected["replaced_f1"]:.3f}', 17, '#26765d')
    label(.055, .163, f'내보낸 모델(frozen-encoder): {result["selected_epoch"]} epoch (검증 토큰 판별 손실이 가장 낮은 epoch)', 11, '#68778b')
    label(.055, .11, '모든 epoch를 동일한 치환 문제로 비교했습니다.', 10, '#68778b')
    label(.055, .077, '측정 대상은 토큰 치환 판별입니다. 감성·속성 판정은 뒤 단계(LoRA + 헤드)에서 따로 평가합니다.', 10, '#68778b')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in text:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    fig.savefig(OUT / 'training-curve.png', facecolor='white')
    plt.close(fig)
    print('wrote', OUT / 'training-curve.png')


if __name__ == '__main__':
    main()
