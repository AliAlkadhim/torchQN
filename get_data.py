import data_loader_two_by_two as dat
# import exer_2_solution as dat
train_generator, eval_generator = dat.get_data_set()
print(next(train_generator()))
print(next(eval_generator()))