# Pretrain
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Art  --phase pretrain --gpu 0 --lambda_div 0.1 --epochs 20
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Clipart  --phase pretrain --gpu 0 --lambda_div 0.1 --epochs 20
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Product  --phase pretrain --gpu 0 --lambda_div 0.1 --epochs 20
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Real  --phase pretrain --gpu 0 --lambda_div 0.1 --epochs 20

# Main training
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Art    --target Product   --phase train --gpu 0  --steps 70000 --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Art    --target Real   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Art --target Clipart   --phase train --gpu 0  --steps 70000 --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Clipart    --target Art   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Clipart    --target Product   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Clipart    --target Real   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Product    --target Art   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Product    --target Clipart   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Product    --target Real   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Real    --target Art   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Real    --target Clipart   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Real    --target Product   --phase train --gpu 0  --steps 70000  --lambda_div 0.1
