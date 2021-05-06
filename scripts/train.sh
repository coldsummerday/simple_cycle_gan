python3 -u  tools/testlmdbmodel.py --source_lmdb ./data/lmdbdataset/src_generate \
    --target_lmdb ./data/lmdbdataset/train_1w_zhb --test_lmdb ./data/lmdbdataset/val_1w_zhb \
    --name zhb_cycle_ocr_cond  --model cycle_ocr_cond \
    --gpu_ids 0  --direction AtoB  --display_port 8008  --display_id=0 \
    --netG unet_32 \
    --netD n_spect_layers  --n_layers_D 1  \
    --dataset_mode shuffle \
    --pool_size 100 \
    --lr 0.0002  --niter 20  --niter_decay 10 \
    --lambda_identity 10.0 \
    --preprocess scale_width_and_crop  \
    --display_freq 1000 --update_html_freq 1000 \
    --load_size 32 --crop_size 32 640 \
    --print_freq 100 --save_epoch_freq 5 \
    --pretrain_ctc_model ./checkpoints/train_srcg/ocr-44000.pth \
    --num_threads 8 --batch_size 1

python3 tools/train.py configs/cycle_ocr_cond/cycle_ocr_cond_ctc.py