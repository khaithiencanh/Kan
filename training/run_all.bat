@echo off
set MODEL_PATH="C:\Users\Administrator\.cache\modelscope\hub\models\AI-ModelScope\stable-diffusion-v2-1"

echo Bat dau chay tiep cac class con lai...

:: Vong lap tu class 1 den class 14 (vi class 0 da xong)
for /L %%i in (1, 1, 14) do (
    echo.
    echo ==================================================
    echo DANG HUAN LUYEN CLASS %%i
    echo ==================================================
    
    python main.py --pretrained_model_name_or_path=%MODEL_PATH% --dataset=matek --n_template=1 --fewshot_seed=seed0 --train_batch_size=8 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_scheduler=cosine --lr_warmup_steps=100 --num_train_epochs=300 --mixed_precision=fp16 --train_text_encoder=True --is_tqdm=True --output_dir=output --n_shot=16 --target_class_idx=%%i --resume_from_checkpoint=None
)

echo Hoan thanh toan bo 15 class!
pause