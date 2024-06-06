from spl_training_script import train_np_baseline, train_spl_curriculum


for i in range(5):
    train_np_baseline(dataset_key_int=i, dataloader_key_int=i, dataset_size=128*1000, training_step_number=6000, eval_dataset_size=128*100, eval_intervals=500, sampler_ratios=[0.0,1.0], chunk_size=128*100, save_path="./exp_ds_5runs/noise_0_1_runs/",model_name="base_0_1_"+str(i))
    train_spl_curriculum(dataset_key_int=i, dataloader_key_int=i, dataset_size= 128*1000, training_step_number=6000 ,eval_dataset_size=128*100, eval_intervals=500, sampler_ratios=[0.0,1.0], chunk_size=128*100, save_path="./exp_ds_5runs/noise_0_1_runs/",model_name="spl_0_1_"+str(i), start_rate=0.1, growth_epochs=5)

for i in range(5,10):
    train_np_baseline(dataset_key_int=i, dataloader_key_int=i, dataset_size=128*1000, training_step_number=6000, eval_dataset_size=128*100, eval_intervals=500, sampler_ratios=[0.3,0.7], chunk_size=128*100, save_path="./exp_ds_5runs/noise_0.3_0.7_runs/",model_name="base_0.3_0.7_"+str(i))
    train_spl_curriculum(dataset_key_int=i, dataloader_key_int=i, dataset_size= 128*1000, training_step_number=6000 ,eval_dataset_size=128*100, eval_intervals=500, sampler_ratios=[0.3,0.7], chunk_size=128*100, save_path="./exp_ds_5runs/noise_0.3_0.7_runs/",model_name="spl_0.3_0.7_"+str(i), start_rate=0.1, growth_epochs=5)

for i in range(10,15):
    train_np_baseline(dataset_key_int=i, dataloader_key_int=i, dataset_size=128*1000, training_step_number=6000, eval_dataset_size=128*100, eval_intervals=500, sampler_ratios=[0.3,0.7], chunk_size=128*100, save_path="./exp_ds_5runs/noise_0.6_0.4_runs/",model_name="base_0.6_0.4_"+str(i))
    train_spl_curriculum(dataset_key_int=i, dataloader_key_int=i, dataset_size= 128*1000, training_step_number=6000 ,eval_dataset_size=128*100, eval_intervals=500, sampler_ratios=[0.3,0.7], chunk_size=128*100, save_path="./exp_ds_5runs/noise_0.6_0.4_runs/",model_name="spl_0.6_0.4_"+str(i), start_rate=0.1, growth_epochs=5)


