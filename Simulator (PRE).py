# -*- coding: utf-8 -*-
# Copyright (C) 2020 by
# Zhen Su <zhensu@pik-potsdam.de>
# All rights reserved
# GNU General Public License v3.0

import networkx as nx
import Network as net
import numpy as np
from random import randint, uniform, sample, shuffle, random


def single_Hom_SI_NC(p_n, p_a, p_ts, p_shc, p_slc):
    """return node states after simulation
    :param p_n:
    network
    :param p_a:
    infection(spreading) probability
    :param p_ts:
    time steps to iterate
    :param p_shc:
    high centrality sources
    :param p_slc:
    low centrality sources
    :return:
    node states after simulation
    """

    g = nx.Graph(p_n)
    # diffusion source
    S_HC = set()
    I_HC = set()
    S_LC = set()
    I_LC = set()
    for i in g:
        if i in p_shc:
            I_HC.add(i)
        else:
            S_HC.add(i)
        if i in p_slc:
            I_LC.add(i)
        else:
            S_LC.add(i)

    t_sus_hc = np.array([0] * (p_ts + 1))
    t_inf_hc = np.array([0] * (p_ts + 1))
    t_sus_lc = np.array([0] * (p_ts + 1))
    t_inf_lc = np.array([0] * (p_ts + 1))

    # start simulating
    new_I_HC = set()
    new_I_LC = set()
    for t in range(1, p_ts + 1):
        new_I_HC.clear()
        for i in S_HC:
            for adj_i in g[i]:
                if (adj_i in I_HC) and (random() <= p_a):
                    new_I_HC.add(i)
                    break
        new_I_LC.clear()
        for i in S_LC:
            for adj_i in g[i]:
                if (adj_i in I_LC) and (random() <= p_a):
                    new_I_LC.add(i)
                    break

        for new_i in new_I_HC:
            S_HC.remove(new_i)
            I_HC.add(new_i)
        for new_i in new_I_LC:
            S_LC.remove(new_i)
            I_LC.add(new_i)

        t_sus_hc[t] = len(S_HC)
        t_inf_hc[t] = len(I_HC)
        t_sus_lc[t] = len(S_LC)
        t_inf_lc[t] = len(I_LC)
    return t_sus_hc, t_inf_hc, t_sus_lc, t_inf_lc


def single_Hom_SIS_NC(p_n, p_a, p_b, p_ts, p_shc, p_slc):
    """return node states after simulation
    :param p_n:
    network
    :param p_a:
    infection(spreading) probability
    :param p_b:
    resusceptible probability for SIS
    :param p_ts:
    time steps to iterate
    :param p_shc:
    high centrality sources
    :param p_slc:
    low centrality sources
    :return:
    node states after simulation
    """

    g = nx.Graph(p_n)
    # diffusion source
    S_HC = set()
    I_HC = set()
    S_LC = set()
    I_LC = set()
    for i in g:
        if i in p_shc:
            I_HC.add(i)
        else:
            S_HC.add(i)
        if i in p_slc:
            I_LC.add(i)
        else:
            S_LC.add(i)

    t_sus_hc = np.array([0] * (p_ts + 1))
    t_inf_hc = np.array([0] * (p_ts + 1))
    t_sus_lc = np.array([0] * (p_ts + 1))
    t_inf_lc = np.array([0] * (p_ts + 1))

    # start simulating
    new_I_HC = set()
    new_I_LC = set()
    for t in range(1, p_ts + 1):
        new_I_HC.clear()
        for i in S_HC:
            for adj_i in g[i]:
                if (adj_i in I_HC) and (random() <= p_a):
                    new_I_HC.add(i)
                    break

        new_I_LC.clear()
        for i in S_LC:
            for adj_i in g[i]:
                if (adj_i in I_LC) and (random() <= p_a):
                    new_I_LC.add(i)
                    break

        temp_I_HC = I_HC.copy()
        temp_I_LC = I_LC.copy()
        for i in temp_I_HC:
            if random() <= p_b:
                S_HC.add(i)
                I_HC.remove(i)
        for i in temp_I_LC:
            if random() <= p_b:
                S_LC.add(i)
                I_LC.remove(i)
        for new_i in new_I_HC:
            S_HC.remove(new_i)
            I_HC.add(new_i)
        for new_i in new_I_LC:
            S_LC.remove(new_i)
            I_LC.add(new_i)

        t_sus_hc[t] = len(S_HC)
        t_inf_hc[t] = len(I_HC)
        t_sus_lc[t] = len(S_LC)
        t_inf_lc[t] = len(I_LC)
    return t_sus_hc, t_inf_hc, t_sus_lc, t_inf_lc


def single_Hom_SIR_NC(p_n, p_a, p_b, p_ts, p_shc, p_slc):
    """return node states after simulation
    :param p_n:
    network
    :param p_a:
    infection(spreading) probability
    :param p_b:
    recovery probability for SIR
    :param p_ts:
    time steps to iterate
    :param p_shc:
    high centrality sources
    :param p_slc:
    low centrality sources
    :return:
    node states after simulation
    """

    g = nx.Graph(p_n)
    # diffusion source
    S_HC = set()
    I_HC = set()
    R_HC = set()
    S_LC = set()
    I_LC = set()
    R_LC = set()
    for i in g:
        if i in p_shc:
            I_HC.add(i)
        else:
            S_HC.add(i)
        if i in p_slc:
            I_LC.add(i)
        else:
            S_LC.add(i)

    t_sus_hc = np.array([0] * (p_ts + 1))
    t_inf_hc = np.array([0] * (p_ts + 1))
    t_rec_hc = np.array([0] * (p_ts + 1))
    t_sus_lc = np.array([0] * (p_ts + 1))
    t_inf_lc = np.array([0] * (p_ts + 1))
    t_rec_lc = np.array([0] * (p_ts + 1))

    # start simulating
    new_I_HC = set()
    new_I_LC = set()
    for t in range(1, p_ts + 1):
        new_I_HC.clear()
        for i in S_HC:
            for adj_i in g[i]:
                if (adj_i in I_HC) and (random() <= p_a):
                    new_I_HC.add(i)
                    break
        new_I_LC.clear()
        for i in S_LC:
            for adj_i in g[i]:
                if (adj_i in I_LC) and (random() <= p_a):
                    new_I_LC.add(i)
                    break

        temp_I_HC = I_HC.copy()
        temp_I_LC = I_LC.copy()
        for i in temp_I_HC:
            if random() <= p_b:
                R_HC.add(i)
                I_HC.remove(i)
        for i in temp_I_LC:
            if random() <= p_b:
                R_LC.add(i)
                I_LC.remove(i)
        for new_i in new_I_HC:
            S_HC.remove(new_i)
            I_HC.add(new_i)
        for new_i in new_I_LC:
            S_LC.remove(new_i)
            I_LC.add(new_i)

        t_sus_hc[t] = len(S_HC)
        t_inf_hc[t] = len(I_HC)
        t_rec_hc[t] = len(R_HC)
        t_sus_lc[t] = len(S_LC)
        t_inf_lc[t] = len(I_LC)
        t_rec_lc[t] = len(R_LC)
    return t_sus_hc, t_inf_hc, t_rec_hc, t_sus_lc, t_inf_lc, t_rec_lc


def single_Hom_IC_NC(p_n, p_a, p_ts, p_shc, p_slc):
    """return node states after simulation
    :param p_n:
    network
    :param p_a:
    infection(spreading) probability
    :param p_ts:
    time steps to iterate
    :param p_shc:
    high centrality sources
    :param p_slc:
    low centrality sources
    :return:
    node states after simulation
    """

    g = nx.Graph(p_n)
    # diffusion source
    I_HC = set()
    A_HC = set()
    CA_HC = set()
    I_LC = set()
    A_LC = set()
    CA_LC = set()
    for i in g:
        if i in p_shc:
            A_HC.add(i)
            CA_HC.add(i)
        else:
            I_HC.add(i)
        if i in p_slc:
            A_LC.add(i)
            CA_LC.add(i)
        else:
            I_LC.add(i)

    t_ina_hc = {}
    t_act_hc = {}
    t_ca_hc = {}
    t_ina_lc = {}
    t_act_lc = {}
    t_ca_lc = {}

    # start simulating
    new_A_HC = set()
    new_A_LC = set()
    for t in range(1, p_ts + 1):
        new_A_HC.clear()
        for i in CA_HC:
            for adj_i in g[i]:
                if (adj_i in I_HC) and (random() <= p_a):
                    new_A_HC.add(adj_i)
        new_A_LC.clear()
        for i in CA_LC:
            for adj_i in g[i]:
                if (adj_i in I_LC) and (random() <= p_a):
                    new_A_LC.add(adj_i)

        CA_HC.clear()
        for new_a_hc in new_A_HC:
            I_HC.remove(new_a_hc)
            A_HC.add(new_a_hc)
            CA_HC.add(new_a_hc)
        CA_LC.clear()
        for new_a_lc in new_A_LC:
            I_LC.remove(new_a_lc)
            A_LC.add(new_a_lc)
            CA_LC.add(new_a_lc)

        t_ina_hc[t] = len(I_HC)
        t_act_hc[t] = len(A_HC)
        t_ca_hc[t] = len(CA_HC)
        t_ina_lc[t] = len(I_LC)
        t_act_lc[t] = len(A_LC)
        t_ca_lc[t] = len(CA_LC)
    return t_ina_hc, t_act_hc, t_ca_hc, t_ina_lc, t_act_lc, t_ca_lc
