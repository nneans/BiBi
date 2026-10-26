"""Validate gold-300.json: evidence must appear in the review text, values must be allowed."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'gold-300.json'
    doc = json.loads(path.read_text(encoding='utf-8'))
    guide = doc['_guide']
    aspects, sentiments, targets = set(guide['속성']), set(guide['감성']), set(guide['대상'])
    errors, done, opinions = [], 0, 0
    for r in doc['reviews']:
        where = f"no {r['no']} ({r['review_id']})"
        if r['done'] is True:
            done += 1
        elif r['done'] is not False:
            errors.append(f'{where}: done은 true 또는 false')
        for k, o in enumerate(r['opinions'], 1):
            opinions += 1
            ev = o.get('evidence', '')
            if not ev:
                errors.append(f'{where} 의견{k}: evidence가 비어 있음')
            elif ev not in r['text']:
                errors.append(f'{where} 의견{k}: evidence "{ev}"가 원문에 없음')
            if o.get('aspect') not in aspects:
                errors.append(f"{where} 의견{k}: 속성 \"{o.get('aspect')}\"는 목록에 없음")
            if o.get('sentiment') not in sentiments:
                errors.append(f"{where} 의견{k}: 감성 \"{o.get('sentiment')}\"는 목록에 없음")
            if o.get('target', '이 상품') not in targets:
                errors.append(f"{where} 의견{k}: 대상 \"{o.get('target')}\"는 목록에 없음")
            if o.get('aspect') == '사실' and o.get('sentiment') != '해당없음':
                errors.append(f'{where} 의견{k}: 사실은 감성을 "해당없음"으로')
        if r['opinions'] and r['done'] is False:
            pass
    print(f"완료 {done}/{len(doc['reviews'])}건, 의견 {opinions}개, 오류 {len(errors)}개")
    for e in errors:
        print(' -', e)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
