### install

```bash
poetry install
```


### modelのダウンロード

```bash
poetry run huggingface-cli download juliozhao/DocLayout-YOLO-DocStructBench --local-dir ./models/DocLayout-YOLO-DocStructBench
```


### jpg filesの設置

input_jpg_filesに解析対象のjpgを設置


### 解析実行

```bash
poetry run python create_json.py
```
