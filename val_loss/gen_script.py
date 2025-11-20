
iter_nunms = list(range(10000, 540000, 2000))
iter_nunms = reversed(iter_nunms)
# print(list(iter_nunms))

print("printf 'model_optim_rng.pt\\xe2\\x80\\x8b' > /tmp/key.zwsp")
print()

for iter_num in iter_nunms:
    iter_num_str = f'{iter_num:07d}'
    if iter_num % 10000 != 0:
        print(f"ks3util cp ks3://xd-model/i_line_ckpt/i_line_s1_fp8_0921/iter_{iter_num_str}/mp_rank_00/$(cat /tmp/key.zwsp) /data2/liuyang/i_line_ckpt/i_line_s1_fp8_0921/iter_{iter_num_str}/mp_rank_00/model_optim_rng.pt -f")
