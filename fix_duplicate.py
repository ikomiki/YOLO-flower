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
            skip_duplicate = False
            
            while i < len(lines):
                line = lines[i]
                
                # label_linesの重複を検出
                if 'label_lines = [' in line and not skip_duplicate:
                    # 次の数行を確認して、同じパターンがあるかチェック
                    j = i + 1
                    found_duplicate = False
                    while j < len(lines) and j < i + 10:
                        if 'label_lines = [' in lines[j]:
                            found_duplicate = True
                            break
                        if 'with open(label_path' in lines[j]:
                            break
                        j += 1
                    
                    if found_duplicate:
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
                        
                        # 重複している部分をスキップ
                        while i < len(lines) and 'label_lines = [' in lines[i]:
                            # コメント行もスキップ
                            while i < len(lines) and ('# Python側で文字列を結合' in lines[i] or 'label_lines = [' in lines[i] or (']' not in lines[i] and 'with open' not in lines[i])):
                                i += 1
                            if i < len(lines) and ']' in lines[i]:
                                i += 1  # ]の行をスキップ
                            break
                        continue
                    else:
                        new_lines.append(line)
                        i += 1
                        continue
                
                # val_count += 1をtrain_count += 1に修正（train用の処理内）
                if 'val_count += 1' in line and 'train_labels_dir' in ''.join(new_lines[-20:]):
                    new_lines.append('                train_count += 1\n')
                    i += 1
                    continue
                
                new_lines.append(line)
                i += 1
            
            if len(new_lines) != len(lines):
                cell['source'] = new_lines
                with open('YOLO-flower.ipynb', 'w', encoding='utf-8') as f:
                    json.dump(nb, f, ensure_ascii=False, indent=1)
                print(f"修正完了: {len(lines)}行 -> {len(new_lines)}行")
            else:
                print("変更なし")
            break

