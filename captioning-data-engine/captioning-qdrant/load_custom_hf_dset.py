from datasets import load_from_disk

save_directory = "custom_hf_dset"
dset_loaded = load_from_disk(save_directory)
print(dset_loaded)

df = dset_loaded.to_pandas()
print(df.head(8))
print(df['image_path'][0:20])