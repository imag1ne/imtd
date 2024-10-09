def discover_petri_net_inductive_bi(logp, logm, parameters=None, sup=None, ratio=None, size_par=None, parallel=False):
    from imtd.algo.discovery.inductive.variants.im_bi import algorithm as im_bi_algo
    return im_bi_algo.apply(logp, logm, parameters=parameters, sup=sup, ratio=ratio, size_par=size_par,
                            parallel=parallel)


def discover_petri_net_inductive_td(logp, logm, similarity_matrix, parameters=None, sup=None, ratio=None,
                                    size_par=None, weight=None):
    from imtd.algo.discovery.inductive.variants.im_td import algorithm as im_td_algo
    return im_td_algo.apply(logp, logm, similarity_matrix, parameters=parameters, sup=sup, ratio=ratio,
                            size_par=size_par, weight=weight)


def discover_petri_net_inductive(logp, logm, weight=0.0, noise_threshold=0.0, parameters=None):
    from imtd.algo.discovery.inductive.variants.im_f import algorithm as imf_algo
    return imf_algo.apply(logp, logm, weight, noise_threshold=noise_threshold, parameters=parameters)
