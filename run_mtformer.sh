
docker run -dit --name run_mtformer  --gpus all --shm-size 256g \
-v /{target_folder}:/workspace \
pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
docker attach run_mtformer


pip install -r MTFormer/requirements.txt

python MTFormer/main.py \
-t mtformer --result_dir "logs/derma_c3_0" \
--att_h 3 --att_l 0 1 2 --num_layer 3 \
--input_size 32 -e_svdd 100 --in_channels 3 \
--data_dir "data/derma" -n 1 2 3 4 5 6 -an 0

python MTFormer/main.py \
-t mtformer --result_dir "logs/derma_c3_1" \
--att_h 3 --att_l 0 1 2 --num_layer 3 \
--input_size 32 -e_svdd 100 --in_channels 3 \
--data_dir "data/derma" -n 0 2 3 4 5 6 -an 1

python MTFormer/main.py \
-t mtformer --result_dir "logs/derma_c3_2" \
--att_h 3 --att_l 0 1 2 --num_layer 3 \
--input_size 32 -e_svdd 100 --in_channels 3 \
--data_dir "data/derma" -n 0 1 3 4 5 6 -an 2

python MTFormer/main.py \
-t mtformer --result_dir "logs/derma_c3_3" \
--att_h 3 --att_l 0 1 2 --num_layer 3 \
--input_size 32 -e_svdd 100 --in_channels 3 \
--data_dir "data/derma" -n 0 1 2 4 5 6 -an 3

python MTFormer/main.py \
-t mtformer --result_dir "logs/derma_c3_4" \
--att_h 3 --att_l 0 1 2 --num_layer 3 \
--input_size 32 -e_svdd 100 --in_channels 3 \
--data_dir "data/derma" -n 0 1 2 3 5 6 -an 4

python MTFormer/main.py \
-t mtformer --result_dir "logs/derma_c3_5" \
--att_h 3 --att_l 0 1 2 --num_layer 3 \
--input_size 32 -e_svdd 100 --in_channels 3 \
--data_dir "data/derma" -n 0 1 2 3 4 6 -an 5

python MTFormer/main.py \
-t mtformer --result_dir "logs/derma_c3_6" \
--att_h 3 --att_l 0 1 2 --num_layer 3 \
--input_size 32 -e_svdd 100 --in_channels 3 \
--data_dir "data/derma" -n 0 1 2 3 4 5 -an 6
