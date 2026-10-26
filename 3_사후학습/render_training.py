"""Render the actual Stage 1 fixed-corruption learning curve."""
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
    label(.055, .96, 'Stage 1 · 식품 리뷰 추가 학습', 23)
    label(.055, .906, f'학습 원문 50,000건 · {result["training_products"]}개 제품 · 3 epoch 완료', 13)
    label(.055, .87, '검증 2,000건 · 최종 평가 2,000건 / 리뷰 원문 전체를 입력', 11, '#68778b')
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
    label(.055, .163, f'선택한 모델: {result["selected_epoch"]} epoch · 생성기 준비와 저장 포함 {result["elapsed_seconds_including_generator_bootstrap_and_model_saving"] / 60:.1f}분', 11, '#68778b')
    label(.055, .11, '모든 epoch를 동일한 치환 문제로 비교했습니다.', 10, '#68778b')
    label(.055, .077, '측정 대상은 토큰 치환 판별입니다. 감성·대상 연결 head의 학습과 평가는 Stage 2에서 진행합니다.', 10, '#68778b')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in text:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    fig.savefig(OUT / 'stage1-training.png', facecolor='white')
    plt.close(fig)
    (OUT / 'visual-verification.json').write_text(json.dumps({'source': 'results.json and history.json',
        'fixed_corruption_validation_points': 4, 'test_before_after_points': 2,
        'all_figure_text_within_bounds': True, 'metric_is_semantic_task_accuracy': False}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
