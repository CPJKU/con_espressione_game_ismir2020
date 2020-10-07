from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times'], 'size' : 15})
rc('text', usetex=True)
import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import scipy.stats as stats

from stat_models import f_oneway, ttest_ind

if __name__ == '__main__':

    results_dir = '../text_analysis_data'
    similarity_fn = os.path.join(results_dir, 'answer_similarity.csv')

    if not os.path.exists(similarity_fn):
        results_filenames = glob.glob(os.path.join(results_dir, 'semantic_similarity*.csv'))

        fptrn = re.compile('semantic_similarity_([0-9]+).csv')

        results_files = dict()
        folds = []
        for i, filename in enumerate(results_filenames):
            fold = int(fptrn.search(filename).group(1))
            df = pd.read_csv(filename)
            df.set_index('indices_k')
            folds.append(df)

        similarity = pd.concat(folds, axis=0)
        similarity.sort_values(by=['indices_k'], inplace=True)
        similarity.set_index('indices_k')

        # correct assignment of music_ids
        df = pd.read_csv("../data/biggame.csv")
        similarity['music_id_i'] = df['music_id'][similarity['indices_i'].values].values
        similarity['music_id_j'] = df['music_id'][similarity['indices_j'].values].values
        del similarity['Unnamed: 0']

        similarity.to_csv(similarity_fn)

    else:
        similarity = pd.read_csv(similarity_fn)
        similarity.set_index('indices_k')
        del similarity['Unnamed: 0']

    piece_performer_info = pd.read_csv('../data/piece_performer_data.csv')

    music_ids = np.unique(piece_performer_info['music_id'].values)
    piece_names = np.unique(piece_performer_info['piece_name'])
    piece_name_idxs = [np.where(piece_performer_info['piece_name'] == pn)[0]
                       for pn in piece_names]

    music_ids_per_piece = [piece_performer_info['music_id'].values[pix] for pix in piece_name_idxs]

    performance_intra_similarity = np.zeros((len(music_ids), 2))
    performance_inter_similarity = np.zeros((len(music_ids), 2))
    for i, mid in enumerate(music_ids):
        inter_idxs = np.where(np.logical_and(similarity['music_id_i'] ==  mid,
                                        similarity['music_id_j'] != mid))[0]
        intra_idxs = np.where(np.logical_and(similarity['music_id_i'] == mid,
                                             similarity['music_id_j'] == mid))[0]
        sim_inter = similarity['answer_similarity'][inter_idxs]
        sim_intra = similarity['answer_similarity'][intra_idxs]
        performance_inter_similarity[i] = (sim_inter.mean(), sim_inter.std())
        performance_intra_similarity[i] = (sim_intra.mean(), sim_intra.std())


    test_performance_similarity = ttest_ind(performance_inter_similarity[:, 0],
                                                  performance_intra_similarity[:, 0])

    print(test_performance_similarity)

    performance_per_piece_intra_similarity = []
    performance_per_piece_inter_similarity = []
    performance_outer_pieces_similarity = []
    most_similar_performances = dict()

    for i, midp in enumerate(music_ids_per_piece):

        for mid in midp:

            perf_in_piece_j = np.isin(similarity['music_id_j'].values, midp[midp != mid])
            # perf_in_piece_j = np.array([(mi in midp and mi != mid) for mi in similarity['music_id_j'].values])

            intra_idxs = np.where(np.logical_and(similarity['music_id_i'] == mid,
                                                 similarity['music_id_j'] == mid))[0]

            inter_idxs = np.where(np.logical_and(similarity['music_id_i'] == mid,
                                                 perf_in_piece_j))[0]
            #
            outer_idxs = np.where(np.logical_and(similarity['music_id_i'] == mid,
                                                 ~np.isin(similarity['music_id_j'].values, midp)))[0]
            sim_inter = similarity['answer_similarity'][inter_idxs]
            sim_intra = similarity['answer_similarity'][intra_idxs]
            sim_outer = similarity['answer_similarity'][outer_idxs]
            # print(ttest_ind(sim_intra, sim_inter))
            inter_mids = np.unique(similarity['music_id_j'][inter_idxs])
            imid_sim = np.zeros(len(inter_mids))
            for ii, imid in enumerate(inter_mids):
                imid_idx = np.where(similarity['music_id_j'][inter_idxs] == imid)[0]
                imid_sim[ii] = sim_inter.values[imid_idx].mean()
            most_similar_performances[str(mid)] = np.column_stack(
                (inter_mids[imid_sim.argsort()[::-1]],
                 imid_sim[imid_sim.argsort()[::-1]]))
            performance_per_piece_inter_similarity.append((i, mid, sim_inter.mean(), sim_inter.std()))
            performance_per_piece_intra_similarity.append((i, mid, sim_intra.mean(), sim_intra.std()))
            performance_outer_pieces_similarity.append((i, mid, sim_outer.mean(), sim_outer.std()))

    performance_per_piece_inter_similarity = np.array(performance_per_piece_inter_similarity)
    performance_per_piece_intra_similarity = np.array(performance_per_piece_intra_similarity)
    performance_outer_pieces_similarity = np.array(performance_outer_pieces_similarity)

    np.savez_compressed('../text_analysis_data/most_similar_performances.npz',
                        **most_similar_performances)

    data = [performance_per_piece_intra_similarity[:, 2] * 100,
            performance_per_piece_inter_similarity[:, 2] * 100,
            performance_outer_pieces_similarity[:, 2] * 100
            ]

    print('inter_perf', ttest_ind(data[0], data[1]))
    print('intra_piece', ttest_ind(data[0], data[2]))
    print('inter-inter', ttest_ind(data[1], data[2]))
    pos = [1, 2, 3]

    anova_similarity = f_oneway(*data)

    print(anova_similarity)

    fig, ax = plt.subplots()
    violinplot = ax.violinplot(data, pos, showmeans=True,
                               showextrema=True)

    for pc in violinplot['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('black')
        pc.set_linewidth(0)
        pc.set_alpha(0.5)

    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violinplot[partname]
        vp.set_edgecolor('firebrick')
        vp.set_linewidth(0.9)
        vp.set_alpha(0.7)

    for partname in ('cmeans', ):
        vp = violinplot[partname]
        vp.set_edgecolor('firebrick')
        vp.set_linewidth(1.5)
        vp.set_alpha(1)


    ax.set_xticks(pos)
    ax.set_xticklabels(['Intra-performance',
                        'Intra-piece',
                        'Inter-piece'])
    ax.set_ylabel('Semantic Similarity (\%)')
    ax.set_yticks([10, 15, 20])
    ax.set_ylim((9, 21))
    plt.tight_layout()
    plt.savefig('../text_analysis_data/answer_similarity.pdf')
    # plt.show()

        
        





    
