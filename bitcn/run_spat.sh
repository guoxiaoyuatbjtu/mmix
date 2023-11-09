CUDA_VISIBLE_DEVICES=2 python main.py --epochs 500 --input_size 170 --batch_size 256 --seq_len 48 --hid_size 128 --data PEMS08 --pattern PatternB --aug spat 
# python main.py --epochs 500 --input_size 325 --batch_size 256 --seq_len 48 --hid_size 128 --data PEMS-BAY --pattern PatternB --aug spat 
# python main.py --epochs 500 --input_size 323 --batch_size 256 --seq_len 48 --hid_size 128 --data SeattleCycle --pattern PatternB --aug spat 
CUDA_VISIBLE_DEVICES=2 python main.py --epochs 500 --input_size 170 --batch_size 256 --seq_len 48 --hid_size 128 --data PEMS08 --pattern PatternC --aug spat 
# python main.py --epochs 500 --input_size 325 --batch_size 256 --seq_len 48 --hid_size 128 --data PEMS-BAY --pattern PatternC --aug spat
# python main.py --epochs 500 --input_size 323 --batch_size 256 --seq_len 48 --hid_size 128 --data SeattleCycle --pattern PatternC --aug spat
