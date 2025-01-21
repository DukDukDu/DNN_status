import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import os,sys
from torch import nn
import subprocess
import time

def train_multi_nn(fc1, fc2, fc3, nl1, nl2, nl3, lr, purity, s_p, group, e_g):
    epochs = 400
    if lr == 0.01:
        epochs = 500
    nb_of_nn_1 = 0
    nb_of_nn_2 = 0
    nb_of_nn_3 = 0
    nb_of_nn_4 = 0
    log_nb = 0
    commands = []
    os.system("rm -rf for_app.sh")
    os.system("rm -rf train.sh")
    for _fc1 in fc1:
        for _fc2 in fc2:
            for _fc3 in fc3:
                for _nl2 in nl2:
                    for _group in group:

                        if _group == '123':
                            with open("train.sh", 'a') as t_file:

                                t_file.write("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                                > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_1, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb) + '\n')
                            commands.append("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                            --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                            > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_1, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb))
                            with open("for_app.sh", 'a') as file:
                                file.write("nohup python application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/apply{0:03d}/{7} \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/output{0:03d}/{7}/test_model_fjmm.pt \
                                --inputcsv /home/olympus/MingxuanZhang/fatjet/{8}/apply{10}_{7}.csv --outputcsv o{0:03d}_{7}.csv > ./log/app{11:03d}.txt \
                                2>&1 &".format(nb_of_nn_1, _fc1, _fc2, _fc3, nl1, _nl2, nl3, s_p, purity, _group, e_g[0], log_nb) + '\n')
                                file.write("sleep 3" + "\n")
                            nb_of_nn_1 = nb_of_nn_1 +1
                        
                        elif _group == '124':
                            with open("train.sh", 'a') as t_file:
                                t_file.write("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                                > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_2, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb) + '\n')
                            commands.append("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                            --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                            > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_2, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb))
                            with open("for_app.sh", 'a') as file:
                                file.write("nohup python application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/apply{0:03d}/{7} \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/output{0:03d}/{7}/test_model_fjmm.pt \
                                --inputcsv /home/olympus/MingxuanZhang/fatjet/{8}/apply{10}_{7}.csv --outputcsv o{0:03d}_{7}.csv > ./log/app{11:03d}.txt \
                                2>&1 &".format(nb_of_nn_2, _fc1, _fc2, _fc3, nl1, _nl2, nl3, s_p, purity, _group, e_g[1], log_nb) + '\n')
                                file.write("sleep 3" + "\n")
                            nb_of_nn_2 = nb_of_nn_2 +1
                        
                        elif _group == '134':
                            with open("train.sh", 'a') as t_file:
                                t_file.write("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                                > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_3, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb) + '\n')
                            commands.append("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                            --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                            > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_3, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb))
                            with open("for_app.sh", 'a') as file:
                                file.write("nohup python application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/apply{0:03d}/{7} \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/output{0:03d}/{7}/test_model_fjmm.pt \
                                --inputcsv /home/olympus/MingxuanZhang/fatjet/{8}/apply{10}_{7}.csv --outputcsv o{0:03d}_{7}.csv > ./log/app{11:03d}.txt \
                                2>&1 &".format(nb_of_nn_3, _fc1, _fc2, _fc3, nl1, _nl2, nl3, s_p, purity, _group, e_g[2], log_nb) + '\n')
                                file.write("sleep 3" + "\n")
                            nb_of_nn_3 = nb_of_nn_3 +1
                        
                        elif _group == '234':
                            with open("train.sh", 'a') as t_file:
                                t_file.write("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                                > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_4, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb) + '\n')
                            commands.append("nohup python hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/{9}/train{11}_{10}.csv \
                            --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --epochs {7} --lr {8} --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{11}/n_out_app/output{0:03d}/{10} \
                            > log/o{12:03d}.txt 2>&1 & ".format(nb_of_nn_4, _fc1, _fc2, _fc3, nl1, _nl2, nl3, epochs, lr, purity, s_p, _group, log_nb))
                            with open("for_app.sh", 'a') as file:
                                file.write("nohup python application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/apply{0:03d}/{7} \
                                --fc1 {1} --fc2 {2} --fc3 {3} --nl1 {4} --nl2 {5} --nl3 {6} --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/{9}/n_out_app/output{0:03d}/{7}/test_model_fjmm.pt \
                                --inputcsv /home/olympus/MingxuanZhang/fatjet/{8}/apply{10}_{7}.csv --outputcsv o{0:03d}_{7}.csv > ./log/app{11:03d}.txt \
                                2>&1 &".format(nb_of_nn_4, _fc1, _fc2, _fc3, nl1, _nl2, nl3, s_p, purity, _group, e_g[3], log_nb) + '\n')
                                file.write("sleep 3" + "\n")
                            nb_of_nn_4 = nb_of_nn_4 +1

                        log_nb += 1
    return commands

def train_multi_new_nn(fcn, lr, purity, s_p, group, e_g):
    nb_nn = len(fcn)
    nb_layers = fcn[0]
    print('number of NN:', nb_nn)
    print('number of layers(sample):', nb_layers)
    commands = []
    log_nb = 0
    os.system("rm -rf train_new.sh")
    os.system("rm -rf for_app_new.sh")
    for i in range(0, nb_nn):
        for _group in group:
            if _group == '123':
                with open("train_new.sh", 'a') as t_file:
                    t_file.write('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train123_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/123/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb)+'\n')
                    commands.append('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train123_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/123/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb))

                with open("for_app_new.sh", 'a') as v_file:
                    v_file.write('nohup python new_application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/123/n_out_app/apply{0:03d}/h \
                    --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/123/n_out_app/output{0:03d}/h/test_model_fjmm.pt \
                    --inputcsv /home/olympus/MingxuanZhang/fatjet/high/apply4_h.csv --outputcsv o{0:03d}_h.csv --int_array {2}\
                    > ./log/app{1:03d}.txt 2>&1 &'.format(i, log_nb, fcn[i])+'\n')
                
            elif _group == '124':
                with open("train_new.sh", 'a') as t_file:
                    t_file.write('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train124_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/124/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb)+'\n')
                    commands.append('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train124_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/124/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb))

                with open("for_app_new.sh", 'a') as v_file:
                    v_file.write('nohup python new_application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/124/n_out_app/apply{0:03d}/h \
                    --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/124/n_out_app/output{0:03d}/h/test_model_fjmm.pt \
                    --inputcsv /home/olympus/MingxuanZhang/fatjet/high/apply3_h.csv --outputcsv o{0:03d}_h.csv --int_array {2}\
                    > ./log/app{1:03d}.txt 2>&1 &'.format(i, log_nb, fcn[i])+'\n')

            elif _group == '134':
                with open("train_new.sh", 'a') as t_file:
                    t_file.write('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train134_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/134/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb)+'\n')
                    commands.append('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train134_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/134/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb))

                with open("for_app_new.sh", 'a') as v_file:
                    v_file.write('nohup python new_application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/134/n_out_app/apply{0:03d}/h \
                    --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/134/n_out_app/output{0:03d}/h/test_model_fjmm.pt \
                    --inputcsv /home/olympus/MingxuanZhang/fatjet/high/apply2_h.csv --outputcsv o{0:03d}_h.csv --int_array {2}\
                    > ./log/app{1:03d}.txt 2>&1 &'.format(i, log_nb, fcn[i])+'\n')

            elif _group == '234':
                with open("train_new.sh", 'a') as t_file:
                    t_file.write('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train234_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/234/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb)+'\n')
                    commands.append('nohup python new_hmm_fjmm.py --traincsv /home/olympus/MingxuanZhang/fatjet/high/train234_h.csv \
                    --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/234/n_out_app/output{0:03d}/h --int_array {1} \
                    --epochs 500 --lr 0.01 > log/o{2:03d}.txt 2>&1 &'.format(i, fcn[i], log_nb))

                with open("for_app_new.sh", 'a') as v_file:
                    v_file.write('nohup python new_application_fjmm.py --outputdir /home/olympus/MingxuanZhang/fatjet/output_div/234/n_out_app/apply{0:03d}/h \
                    --inputmodel /home/olympus/MingxuanZhang/fatjet/output_div/234/n_out_app/output{0:03d}/h/test_model_fjmm.pt \
                    --inputcsv /home/olympus/MingxuanZhang/fatjet/high/apply1_h.csv --outputcsv o{0:03d}_h.csv --int_array {2}\
                    > ./log/app{1:03d}.txt 2>&1 &'.format(i, log_nb, fcn[i])+'\n')

            log_nb = log_nb + 1

# def multi_sub(commands, _time):
#     train_repre = 0
#     i = 0
#     tot = len(commands)
#     os.system("rm -rf ./log/o*.txt")
#     while i < tot:
#         try:
#             if train_repre == 0 :
#                 os.system(commands[i])
#             else:
#                 pass
#             log_filename = './log/o{0:03d}.txt'.format(i)
#             with open(log_filename, 'r') as file:
#                 content = file.read()
    
#                 if 'AUC' in content and 'Save Done!' in content:
#                     os.system(f"echo Found 'AUC' and 'Save Done!' in {log_filename}")
#                     i = i+1
#                     train_repre = 0
#                 else:
#                     os.system(f"echo Waiting for 'AUC' in {log_filename}...")
#                     train_repre = 1

#             if i == tot:
#                 os.system(f"echo All {tot} Dnns have finished !!!")
#                 break

#         except FileNotFoundError:
#             os.system(f"echo File {log_filename} not found. Will check again in 1 minute.")

#         time.sleep(_time)

def main():
    fc1 = [16, 32, 64]
    fc2 = [512, 1024, 2048]
    fc3 = [16, 32, 64]
    fc4 = [512, 1024, 2048]
    fcn = []
    nl1 = [1]
    nl2 = [1, 2, 3]
    nl3 = 1
    lr = 0.01
    purity = 'high'
    s_p = 'h'
    group = ['123', '124', '134', '234']
    e_g = ['4', '3', '2', '1']
    commands = train_multi_nn(fc1, fc2, fc3, nl1, nl2, nl3, lr, purity, s_p, group, e_g)
    for _fc1 in fc1:
        for _fc2 in fc2:
            for _fc3 in fc3:
                for _fc4 in fc4:
                    fcn.append('{0},{1},{3},{2}'.format(_fc1, _fc2, _fc3, _fc4))
    train_multi_new_nn(fcn, lr, purity, s_p, group, e_g)
    # for _nl1 in nl1:
    #     print(_nl1)
    # multi_sub(commands, 180)
    # print(commands[1])

if __name__ == '__main__':
    main()