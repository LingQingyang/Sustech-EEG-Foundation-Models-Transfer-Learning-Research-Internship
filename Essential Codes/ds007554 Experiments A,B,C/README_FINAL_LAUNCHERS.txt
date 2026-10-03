ABC five-state final launcher set
=================================

Canonical code combination
--------------------------
- shared core: /home/lqy/ld_qd_feature_core_v5_0.py
  Filename retained for imports; content is the v5.1.1 covariance hotfix.
- A Python: /home/lqy/run_experiment_A_five_state_ld_qd_tensor_v5_1_1.py
- B Python: /home/lqy/run_experiment_B_five_state_within_global_sfa_v5_1.py
- C Python: /home/lqy/run_experiment_C_slowspace_ld_qd_tensor_v5_1.py
- integrator: /home/lqy/integrate_experiments_ABC_five_state_v5_0.py

Launchers
---------
- run_experiment_A_five_state_ld_qd_tensor_v5_1_1.sh
- run_experiment_B_five_state_within_global_sfa_v5_1.sh
- run_experiment_C_slowspace_ld_qd_tensor_v5_1.sh
- run_experiments_ABC_five_state_integrated_v5_1_1.sh

Default formal root
-------------------
/mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1

The three standalone launchers default to A/, B/, and C/ under the same root.
The integrated launcher delegates to those standalone launchers, then runs the
A/B preflight and strict final integrator. All experiment stages resume by
default.

Typical formal run
------------------
nohup bash /home/lqy/run_experiments_ABC_five_state_integrated_v5_1_1.sh \
  > /mnt/dataset4/yinuo/FM_flow/dataset/expABC_five_state_integrated_v5_1_1/nohup_launcher.log \
  2>&1 < /dev/null &

echo "ABC PID=$!"
disown -a

Typical smoke run
-----------------
SMOKE=1 bash /home/lqy/run_experiments_ABC_five_state_integrated_v5_1_1.sh

Resume only B/C/integration
---------------------------
RUN_A=0 bash /home/lqy/run_experiments_ABC_five_state_integrated_v5_1_1.sh
