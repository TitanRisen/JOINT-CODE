# Code of Joint dIstribution Neural maTching (JOINT)
## Software Environment
Main dependencies versions:
```
python >= 3.7
pytorch = 1.11
```
## Steps
1. Download the public datasets (e.g. Office-Home [1]) and put them in the directory path like "./data". Please refer to the dataset directory in the "./data directory" for the directory placement format. 
2. Move the datasets labels' files in TXT format to the path "./data/list/[DATASET_NAME]" , e.g., "./data/list/officehome", and this path is based on the root path set at the above step.  Add the prefix "labeled_source_images_" to their filename ,e.g., "./data/list/officehome/labeled_source_images_Art.txt". The labels' files are already prepared at "./data/list/" in this directory. Additionally, please provide separate label list files, and refer to the file path as well as the naming method if you want to perform experiments on other datasets);
3. Pre-training step: Run the command at the root of this project dirtory. Note that the default parameter of "net" is "resnet" for ResNet50.
```
   python  main_for_UDA.py --root_dir [datasets_root_path] --dataset [DATASET_NAME] --source [source_name] --phase pretrain --gpu [the_gpu_code] --lambda_div [0 < lambda_div < 1.0] --epochs 20
   ```
   For example, 
   ```
   python  main_for_UDA.py --root_dir ./data --dataset officehome --source Art  --phase pretrain --gpu 0 --lambda_div 0.1 --epochs 20
   ```
  
4. Main traning step: Run the command at the root of this project dirtory
```
python  main_for_UDA.py --root_dir [datasets_root_path]  --dataset [DATASET_NAME]  --source [source_name] --target [target_name] --phase train --gpu [the_gpu_code]  --steps [max_steps] --lambda_div  [0 < lambda_div < 1.0] 
 ```

For example, 
 ```
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Art    --target Product   --phase train --gpu 0  --steps 70000 --lambda_div 0.1
python  main_for_UDA.py --root_dir ./data --dataset officehome --source Art    --target Real   --phase train --gpu 0  --steps 70000 --lambda_div 0.1
 ```

5. Run the command "python main_for_UDA.py -h " for help. If encountering an error like "CUDA out of memory", please adjust the value of "batch_size" (e.g. batch_size=48) in the main code file or consider distributed training.

6. **Run the "examples.sh" to train all the tasks on Office-Home. Refer to the command "bash examples.sh".**
## References
[1] Hemanth, V., Jose, E., Shayok, C., Sethuraman, P.: Deep hashing network for unsupervised domain adaptation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5018–5027 (2017).
