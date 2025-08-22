import src.autojailbreak as ajb

print("AdvBench")
dataset = ajb.read_dataset("advbench")
row = dataset[0]  # Get first row
print(row)

print("HarmBench")
dataset = ajb.read_dataset("harmbench")
row = dataset[0]  # Get first row
print(row)

print("HarmBench Contextual")
dataset = ajb.read_dataset("harmbench-contextual")
row = dataset[0]  # Get first row
print(row)

print("HarmBench Copyright")
dataset = ajb.read_dataset("harmbench-copyright")
row = dataset[0]  # Get first row
print(row)

print("HarmBench Standard")
dataset = ajb.read_dataset("harmbench-standard")
row = dataset[0]  # Get first row
print(row)

print("JBB Harmful")
dataset = ajb.read_dataset("jbb-harmful")
row = dataset[0]  # Get first row
print(row)

print("JBB Benign")
dataset = ajb.read_dataset("jbb-benign")
row = dataset[0]  # Get first row
print(row)

print("JBB All")
dataset = ajb.read_dataset("jbb-all")
row = dataset[0]  # Get first row
print(row)