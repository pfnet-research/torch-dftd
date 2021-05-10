from typing import Dict


def get_dftd3_default_params(
    damping: str = "zero", xc: str = "pbe", tz: bool = False, old: bool = False
) -> Dict[str, float]:
    """Get DFTD3 parameter for specified damping method &  correlational functional.

    Args:
        damping (str): damping method. [zero, bj, zerom, bjm, dftd2] is supported.
        xc (str): exchange-correlation functional.
        tz (bool): Use special parameters for TZ-type calculations.
            Only effective when damping=zero
        old (bool): Use DFT-D2 calculation

    Returns:
        params (Dict): Parameters for s6, rs6, s18, rs18, alp.
    """
    if old:
        assert damping == "zero", "Only zero damping is supported in DFT-D2"
        damping = "dftd2"
    if damping == "bjm":  # version 6 of Original DFTD3
        # s6, rs6, s18, rs18, alp is used.
        s6 = 1.0
        alp = 14.0
        if xc == "b2-plyp":
            rs6 = 0.486434
            s18 = 0.672820
            rs18 = 3.656466
            s6 = 0.640000
        elif xc == "b3-lyp":
            rs6 = 0.278672
            s18 = 1.466677
            rs18 = 4.606311
        elif xc == "b97-d":
            rs6 = 0.240184
            s18 = 1.206988
            rs18 = 3.864426
        elif xc == "b-lyp":
            rs6 = 0.448486
            s18 = 1.875007
            rs18 = 3.610679
        elif xc == "b-p":
            rs6 = 0.821850
            s18 = 3.140281
            rs18 = 2.728151
        elif xc == "pbe":
            rs6 = 0.012092
            s18 = 0.358940
            rs18 = 5.938951
        elif xc == "pbe0":
            rs6 = 0.007912
            s18 = 0.528823
            rs18 = 6.162326
        elif xc == "lc-wpbe":
            rs6 = 0.563761
            s18 = 0.906564
            rs18 = 3.593680
        else:
            raise ValueError(f"[ERROR] Unexpected value xc={xc}")
    elif damping == "zerom":  # version 5
        # s6, rs6, s18, rs18, alp is used.
        s6 = 1.0
        alp = 14.0
        if xc == "b2-plyp":
            rs6 = 1.313134
            s18 = 0.717543
            rs18 = 0.016035
            s6 = 0.640000
        elif xc == "b3-lyp":
            rs6 = 1.338153
            s18 = 1.532981
            rs18 = 0.013988
        elif xc == "b97-d":
            rs6 = 1.151808
            s18 = 1.020078
            rs18 = 0.035964
        elif xc == "b-lyp":
            rs6 = 1.279637
            s18 = 1.841686
            rs18 = 0.014370
        elif xc == "b-p":
            rs6 = 1.233460
            s18 = 1.945174
            rs18 = 0.000000
        elif xc == "pbe":
            rs6 = 2.340218
            s18 = 0.000000
            rs18 = 0.129434
        elif xc == "pbe0":
            rs6 = 2.077949
            s18 = 0.000081
            rs18 = 0.116755
        elif xc == "lc-wpbe":
            rs6 = 1.366361
            s18 = 1.280619
            rs18 = 0.003160
        else:
            raise ValueError(f"[ERROR] Unexpected value xc={xc}")
    elif damping == "bj":
        # version 4, Becke-Johnson finite-damping, variant 2 with their radii
        # s6, rs6, s18, rs18, alp is used.
        s6 = 1.0
        alp = 14.0
        if xc == "b-p":
            rs6 = 0.3946
            s18 = 3.2822
            rs18 = 4.8516
        elif xc == "b-lyp":
            rs6 = 0.4298
            s18 = 2.6996
            rs18 = 4.2359
        elif xc == "revpbe":
            rs6 = 0.5238
            s18 = 2.3550
            rs18 = 3.5016
        elif xc == "rpbe":
            rs6 = 0.1820
            s18 = 0.8318
            rs18 = 4.0094
        elif xc == "b97-d":
            rs6 = 0.5545
            s18 = 2.2609
            rs18 = 3.2297
        elif xc == "pbe":
            rs6 = 0.4289
            s18 = 0.7875
            rs18 = 4.4407
        elif xc == "rpw86-pbe":
            rs6 = 0.4613
            s18 = 1.3845
            rs18 = 4.5062
        elif xc == "b3-lyp":
            rs6 = 0.3981
            s18 = 1.9889
            rs18 = 4.4211
        elif xc == "tpss":
            rs6 = 0.4535
            s18 = 1.9435
            rs18 = 4.4752
        elif xc == "hf":
            rs6 = 0.3385
            s18 = 0.9171
            rs18 = 2.8830
        elif xc == "tpss0":
            rs6 = 0.3768
            s18 = 1.2576
            rs18 = 4.5865
        elif xc == "pbe0":
            rs6 = 0.4145
            s18 = 1.2177
            rs18 = 4.8593
        elif xc == "hse06":
            rs6 = 0.383
            s18 = 2.310
            rs18 = 5.685
        elif xc == "revpbe38":
            rs6 = 0.4309
            s18 = 1.4760
            rs18 = 3.9446
        elif xc == "pw6b95":
            rs6 = 0.2076
            s18 = 0.7257
            rs18 = 6.3750
        elif xc == "b2-plyp":
            rs6 = 0.3065
            s18 = 0.9147
            rs18 = 5.0570
            s6 = 0.64
        elif xc == "dsd-blyp":
            rs6 = 0.0000
            s18 = 0.2130
            rs18 = 6.0519
            s6 = 0.50
        elif xc == "dsd-blyp-fc":
            rs6 = 0.0009
            s18 = 0.2112
            rs18 = 5.9807
            s6 = 0.50
        elif xc == "bop":
            rs6 = 0.4870
            s18 = 3.2950
            rs18 = 3.5043
        elif xc == "mpwlyp":
            rs6 = 0.4831
            s18 = 2.0077
            rs18 = 4.5323
        elif xc == "o-lyp":
            rs6 = 0.5299
            s18 = 2.6205
            rs18 = 2.8065
        elif xc == "pbesol":
            rs6 = 0.4466
            s18 = 2.9491
            rs18 = 6.1742
        elif xc == "bpbe":
            rs6 = 0.4567
            s18 = 4.0728
            rs18 = 4.3908
        elif xc == "opbe":
            rs6 = 0.5512
            s18 = 3.3816
            rs18 = 2.9444
        elif xc == "ssb":
            rs6 = -0.0952
            s18 = -0.1744
            rs18 = 5.2170
        elif xc == "revssb":
            rs6 = 0.4720
            s18 = 0.4389
            rs18 = 4.0986
        elif xc == "otpss":
            rs6 = 0.4634
            s18 = 2.7495
            rs18 = 4.3153
        elif xc == "b3pw91":
            rs6 = 0.4312
            s18 = 2.8524
            rs18 = 4.4693
        elif xc == "bh-lyp":
            rs6 = 0.2793
            s18 = 1.0354
            rs18 = 4.9615
        elif xc == "revpbe0":
            rs6 = 0.4679
            s18 = 1.7588
            rs18 = 3.7619
        elif xc == "tpssh":
            rs6 = 0.4529
            s18 = 2.2382
            rs18 = 4.6550
        elif xc == "mpw1b95":
            rs6 = 0.1955
            s18 = 1.0508
            rs18 = 6.4177
        elif xc == "pwb6k":
            rs6 = 0.1805
            s18 = 0.9383
            rs18 = 7.7627
        elif xc == "b1b95":
            rs6 = 0.2092
            s18 = 1.4507
            rs18 = 5.5545
        elif xc == "bmk":
            rs6 = 0.1940
            s18 = 2.0860
            rs18 = 5.9197
        elif xc == "cam-b3lyp":
            rs6 = 0.3708
            s18 = 2.0674
            rs18 = 5.4743
        elif xc == "lc-wpbe":
            rs6 = 0.3919
            s18 = 1.8541
            rs18 = 5.0897
        elif xc == "b2gp-plyp":
            rs6 = 0.0000
            s18 = 0.2597
            rs18 = 6.3332
            s6 = 0.560
        elif xc == "ptpss":
            rs6 = 0.0000
            s18 = 0.2804
            rs18 = 6.5745
            s6 = 0.750
        elif xc == "pwpb95":
            rs6 = 0.0000
            s18 = 0.2904
            rs18 = 7.3141
            s6 = 0.820
            # special HF/DFT with eBSSE correction
        elif xc == "hf/mixed":
            rs6 = 0.5607
            s18 = 3.9027
            rs18 = 4.5622
        elif xc == "hf/sv":
            rs6 = 0.4249
            s18 = 2.1849
            rs18 = 4.2783
        elif xc == "hf/minis":
            rs6 = 0.1702
            s18 = 0.9841
            rs18 = 3.8506
        elif xc == "b3-lyp/6-31gd":
            rs6 = 0.5014
            s18 = 4.0672
            rs18 = 4.8409
        elif xc == "hcth120":
            rs6 = 0.3563
            s18 = 1.0821
            rs18 = 4.3359
            # DFTB3 old, deprecated parameters:
            #         elif xc == "dftb3":
            #              rs6=0.7461
            #              s18=3.209
            #              rs18=4.1906
            # special SCC-DFTB parametrization
            # full third order DFTB, self consistent charges, hydrogen pair damping with
            #         exponent 4.2
        elif xc == "dftb3":
            rs6 = 0.5719
            s18 = 0.5883
            rs18 = 3.6017
        elif xc == "pw1pw":
            rs6 = 0.3807
            s18 = 2.3363
            rs18 = 5.8844
        elif xc == "pwgga":
            rs6 = 0.2211
            s18 = 2.6910
            rs18 = 6.7278
        elif xc == "hsesol":
            rs6 = 0.4650
            s18 = 2.9215
            rs18 = 6.2003
            # special HF-D3-gCP-SRB/MINIX parametrization
        elif xc == "hf3c":
            rs6 = 0.4171
            s18 = 0.8777
            rs18 = 2.9149
            # special HF-D3-gCP-SRB2/ECP-2G parametrization
        elif xc == "hf3cv":
            rs6 = 0.3063
            s18 = 0.5022
            rs18 = 3.9856
            # special PBEh-D3-gCP/def2-mSVP parametrization
        elif xc in ["pbeh3c", "pbeh-3c"]:
            rs6 = 0.4860
            s18 = 0.0000
            rs18 = 4.5000
        else:
            raise ValueError(f"[ERROR] Unexpected value xc={xc}")
    elif damping == "zero":
        # s6, s18, rs6, rs18, alp is used.
        s6 = 1.0
        rs18 = 1.0
        alp = 14.0
        if not tz:
            if xc == "slater-dirac-exchange":
                rs6 = 0.999
                s18 = -1.957
                rs18 = 0.697
            elif xc == "b-lyp":
                rs6 = 1.094
                s18 = 1.682
            elif xc == "b-p":
                rs6 = 1.139
                s18 = 1.683
            elif xc == "b97-d":
                rs6 = 0.892
                s18 = 0.909
            elif xc == "revpbe":
                rs6 = 0.923
                s18 = 1.010
            elif xc == "pbe":
                rs6 = 1.217
                s18 = 0.722
            elif xc == "pbesol":
                rs6 = 1.345
                s18 = 0.612
            elif xc == "rpw86-pbe":
                rs6 = 1.224
                s18 = 0.901
            elif xc == "rpbe":
                rs6 = 0.872
                s18 = 0.514
            elif xc == "tpss":
                rs6 = 1.166
                s18 = 1.105
            elif xc == "b3-lyp":
                rs6 = 1.261
                s18 = 1.703
            elif xc == "pbe0":
                rs6 = 1.287
                s18 = 0.928

            elif xc == "hse06":
                rs6 = 1.129
                s18 = 0.109
            elif xc == "revpbe38":
                rs6 = 1.021
                s18 = 0.862
            elif xc == "pw6b95":
                rs6 = 1.532
                s18 = 0.862
            elif xc == "tpss0":
                rs6 = 1.252
                s18 = 1.242
            elif xc == "b2-plyp":
                rs6 = 1.427
                s18 = 1.022
                s6 = 0.64
            elif xc == "pwpb95":
                rs6 = 1.557
                s18 = 0.705
                s6 = 0.82
            elif xc == "b2gp-plyp":
                rs6 = 1.586
                s18 = 0.760
                s6 = 0.56
            elif xc == "ptpss":
                rs6 = 1.541
                s18 = 0.879
                s6 = 0.75
            elif xc == "hf":
                rs6 = 1.158
                s18 = 1.746
            elif xc == "mpwlyp":
                rs6 = 1.239
                s18 = 1.098
            elif xc == "bpbe":
                rs6 = 1.087
                s18 = 2.033
            elif xc == "bh-lyp":
                rs6 = 1.370
                s18 = 1.442
            elif xc == "tpssh":
                rs6 = 1.223
                s18 = 1.219
            elif xc == "pwb6k":
                rs6 = 1.660
                s18 = 0.550
            elif xc == "b1b95":
                rs6 = 1.613
                s18 = 1.868
            elif xc == "bop":
                rs6 = 0.929
                s18 = 1.975
            elif xc == "o-lyp":
                rs6 = 0.806
                s18 = 1.764
            elif xc == "o-pbe":
                rs6 = 0.837
                s18 = 2.055
            elif xc == "ssb":
                rs6 = 1.215
                s18 = 0.663
            elif xc == "revssb":
                rs6 = 1.221
                s18 = 0.560
            elif xc == "otpss":
                rs6 = 1.128
                s18 = 1.494
            elif xc == "b3pw91":
                rs6 = 1.176
                s18 = 1.775
            elif xc == "revpbe0":
                rs6 = 0.949
                s18 = 0.792
            elif xc == "pbe38":
                rs6 = 1.333
                s18 = 0.998
            elif xc == "mpw1b95":
                rs6 = 1.605
                s18 = 1.118
            elif xc == "mpwb1k":
                rs6 = 1.671
                s18 = 1.061
            elif xc == "bmk":
                rs6 = 1.931
                s18 = 2.168
            elif xc == "cam-b3lyp":
                rs6 = 1.378
                s18 = 1.217
            elif xc == "lc-wpbe":
                rs6 = 1.355
                s18 = 1.279
            elif xc == "m05":
                rs6 = 1.373
                s18 = 0.595
            elif xc == "m052x":
                rs6 = 1.417
                s18 = 0.000
            elif xc == "m06l":
                rs6 = 1.581
                s18 = 0.000
            elif xc == "m06":
                rs6 = 1.325
                s18 = 0.000
            elif xc == "m062x":
                rs6 = 1.619
                s18 = 0.000
            elif xc == "m06hf":
                rs6 = 1.446
                s18 = 0.000
                # DFTB3 (zeta=4.0), old deprecated parameters
                #         elif xc == "dftb3":
                #              rs6=1.235
                #              s18=0.673
            elif xc == "hcth120":
                rs6 = 1.221
                s18 = 1.206
            else:
                raise ValueError(f"[ERROR] Unexpected value xc={xc}")
        else:
            # special TZVPP parameter
            if xc == "b-lyp":
                rs6 = 1.243
                s18 = 2.022
            elif xc == "b-p":
                rs6 = 1.221
                s18 = 1.838
            elif xc == "b97-d":
                rs6 = 0.921
                s18 = 0.894
            elif xc == "revpbe":
                rs6 = 0.953
                s18 = 0.989
            elif xc == "pbe":
                rs6 = 1.277
                s18 = 0.777
            elif xc == "tpss":
                rs6 = 1.213
                s18 = 1.176
            elif xc == "b3-lyp":
                rs6 = 1.314
                s18 = 1.706
            elif xc == "pbe0":
                rs6 = 1.328
                s18 = 0.926
            elif xc == "pw6b95":
                rs6 = 1.562
                s18 = 0.821
            elif xc == "tpss0":
                rs6 = 1.282
                s18 = 1.250
            elif xc == "b2-plyp":
                rs6 = 1.551
                s18 = 1.109
                s6 = 0.5
            else:
                raise ValueError(f"[ERROR] Unexpected value xc={xc}")
    elif damping == "dftd2":  # version 2, "old=True"
        # s6, rs6, s18, alp is used.
        rs6 = 1.1
        s18 = 0.0
        alp = 20.0
        rs18 = None  # Not used.
        if xc == "b-lyp":
            s6 = 1.2
        elif xc == "b-p":
            s6 = 1.05
        elif xc == "b97-d":
            s6 = 1.25
        elif xc == "revpbe":
            s6 = 1.25
        elif xc == "pbe":
            s6 = 0.75
        elif xc == "tpss":
            s6 = 1.0
        elif xc == "b3-lyp":
            s6 = 1.05
        elif xc == "pbe0":
            s6 = 0.6
        elif xc == "pw6b95":
            s6 = 0.5
        elif xc == "tpss0":
            s6 = 0.85
        elif xc == "b2-plyp":
            s6 = 0.55
        elif xc == "b2gp-plyp":
            s6 = 0.4
        elif xc == "dsd-blyp":
            s6 = 0.41
            alp = 60.0
        else:
            raise ValueError(f"[ERROR] Unexpected value xc={xc}")
    else:
        raise ValueError(f"[ERROR] damping={damping} not supported.")
    return {"s6": s6, "rs6": rs6, "s18": s18, "rs18": rs18, "alp": alp}
