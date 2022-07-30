# training
python -m main --model_name=din \
  --checkpoint_dir=/home/recsys/chapter09/din/checkpoint/ \
  --pattern=/home/recsys/chapter09/din/dataset/* \
  --batch_size=512

# export
python -m lib.serving.export --checkpoint_dir=/home/recsys/chapter09/din/checkpoint \
  --model_dir=/home/recsys/chapter09/din/savers

# serving
docker run -d -p 8501:8501 \
  --mount type=bind,source=/home/recsys/chapter09/din/savers,target=/models/din \
  -e MODEL_NAME=din -t tensorflow/serving

# request: 非紧凑型
curl -X POST \
  http://localhost:8501/v1/models/din:predict \
  -d '{
  "signature_name": "serving_default",
  "instances":[
     {
        "user_id":["user"],
        "age": [18],
        "gender": ["1"],
        "device": ["HuaWei"],
        "item_id": ["item1"],
        "clicks": ["item1","item2","item3"]
     },
     {
        "user_id":["user"],
        "age": [18],
        "gender": ["1"],
        "device": ["HuaWei"],
        "item_id": ["item2"],
        "clicks": ["item1","item2","item3"]
     },
     {
        "user_id":["user"],
        "age": [18],
        "gender": ["1"],
        "device": ["HuaWei"],
        "item_id": ["item3"],
        "clicks": ["item1","item2","item3"]
     }]
  }'

# request: 紧凑型
curl -X POST \
  http://localhost:8501/v1/models/din:predict \
  -d '{
  "signature_name": "serving_default",
  "inputs":
     {
        "user_id":[["user"]],
        "age": [[18]],
        "gender": [["1"]],
        "device": [["HuaWei"]],
        "item_id": [["item1","item2","item3"]],
        "clicks": [["item1","item2","item3"]]
     }
  }'
