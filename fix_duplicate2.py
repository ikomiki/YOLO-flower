import json

with open('YOLO-flower.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'organize_yolo_dataset' in source:
            lines = cell['source']
            new_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i]
                
                # 重複しているlabel_lines定義を検出して修正
                if 'label_lines = [' in line:
                    # 次の数行を確認
                    next_lines = ''.join(lines[i:min(len(lines), i+15)])
                    # 同じブロック内にlabel_linesが2回以上あるか確認
                    if next_lines.count('label_lines = [') > 1:
                        # 最初のlabel_lines定義を追加
                        new_lines.append(line)
                        i += 1
                        # ]までを追加
                        while i < len(lines) and ']' not in lines[i]:
                            new_lines.append(lines[i])
                            i += 1
                        if i < len(lines):
                            new_lines.append(lines[i])  # ]の行
                            i += 1
                        
                        # 重複している部分（コメント + label_lines定義）をスキップ
                        # コメント行をスキップ
                        while i < len(lines) and '# Python側で文字列を結合' in lines[i]:
                            i += 1
                        # label_lines定義をスキップ
                        if i < len(lines) and 'label_lines = [' in lines[i]:
                            while i < len(lines) and ']' not in lines[i]:
                                i += 1
                            if i < len(lines):
                                i += 1  # ]の行をスキップ
                        continue
                
                # val_count += 1をtrain_count += 1に修正（train用の処理内）
                if 'val_count += 1' in line:
                    # 前の20行を確認してtrain_labels_dirがあるかチェック
                    prev_context = ''.join(lines[max(0, i-20):i])
                    if 'train_labels_dir' in prev_context:
                        new_lines.append('                train_count += 1\n')
                        i += 1
                        continue
                
                new_lines.append(line)
                i += 1
            
            if len(new_lines) != len(lines) or any(new_lines[j] != lines[j] for j in range(min(len(new_lines), len(lines)))):
                cell['source'] = new_lines
                with open('YOLO-flower.ipynb', 'w', encoding='utf-8') as f:
                    json.dump(nb, f, ensure_ascii=False, indent=1)
                print(f"修正完了: {len(lines)}行 -> {len(new_lines)}行")
                # 変更内容を確認
                for j in range(min(len(lines), len(new_lines))):
                    if lines[j] != new_lines[j]:
                        print(f"  行 {j}: 変更")
                        print(f"    旧: {lines[j][:80]}")
                        print(f"    新: {new_lines[j][:80]}")
            else:
                print("変更なし")
            break

