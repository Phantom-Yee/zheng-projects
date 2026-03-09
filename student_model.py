from modelscope import snapshot_download

# modelscope/Llama-2-7b-ms
# AI-ModelScope/bloomz-560m
model_name = 'tiansz/bert-base-chinese'
model_dir = snapshot_download(model_id=model_name, # 模型名称
                              cache_dir='./' # 保存在当前根目录
                              )