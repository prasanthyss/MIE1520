import matplotlib.pyplot as plt
import numpy as np

tb_accs = [0.8385337776431384, 0.8324750075734626, 0.8548924568312632, 0.8561042108451984, 0.8673129354740987, 0.868524689488034, 0.8745834595577098, 0.8773099060890639, 0.8782187215995153]
bb_accs = [0.9100272644653136, 0.909421387458346, 0.9148742805210542, 0.9248712511360194, 0.9272947591638897, 0.9363829142684035, 0.9339594062405332, 0.93850348379279, 0.944562253862466]
bl_accs = [0.940018176310209, 0.9542562859739473, 0.960617994547107, 0.9569827325053014, 0.967888518630718, 0.9636473795819449, 0.9697061496516207, 0.9654650106028476, 0.9681914571342017]
rb_accs = [0.9133595880036353, 0.926991820660406, 0.923962435625568, 0.9203271735837625, 0.9263859436534384, 0.928506513177825, 0.9312329597091791, 0.9318388367161466, 0.9342623447440169]
rl_accs = [0.9639503180854286, 0.9633444410784611, 0.9651620720993638, 0.9578915480157528, 0.9687973341411693, 0.9748561042108452, 0.9787943047561345, 0.9706149651620721, 0.9691002726446531]

tb_accs = np.array(tb_accs)
bb_accs = np.array(bb_accs)
bl_accs = np.array(bl_accs)
rb_accs = np.array(rb_accs)
rl_accs = np.array(rl_accs)

tb_accs = (tb_accs - tb_accs[0])/tb_accs[0]
bb_accs = (bb_accs - bb_accs[0])/bb_accs[0]
bl_accs = (bl_accs - bl_accs[0])/bl_accs[0]
rb_accs = (rb_accs - rb_accs[0])/rb_accs[0]
rl_accs = (rl_accs - rl_accs[0])/rl_accs[0]

ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]

plt.plot(ranks, tb_accs, label='TinyBERT', linestyle='-', color='blue', marker='o')
plt.plot(ranks, bb_accs, label='BERT-Base', linestyle='--', color='orange', marker='s')
plt.plot(ranks, rb_accs, label='RoBERTa-Base', linestyle='-.', color='green', marker='^')
plt.plot(ranks, bl_accs, label='BERT-Large', linestyle=':', color='red', marker='d')
plt.plot(ranks, rl_accs, label='RoBERTa-Large', linestyle='-', color='purple', marker='x')

# Add a horizontal dashed line at y = 1
plt.axhline(y=0, color='gray', linestyle='--')

# Set x-axis to log scale base 2
plt.xscale('log', base=2)

# Add labels and title
plt.xlabel('LoRA ranks', fontsize=28)
plt.ylabel('% improvement', fontsize=28)

# Increase the size of the legend
plt.legend(fontsize=24)

# Show plot
plt.show()