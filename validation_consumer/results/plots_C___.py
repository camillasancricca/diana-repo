import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'Times',
        'size'   : 12}

matplotlib.rc('font', **font)

if __name__ == '__main__':

    algorithm = 'AdaBoost'
    lim = [0.7, 0.93]

    i = 0

    # %%
    ### analisi x algoritmo
    # %%

    suggested = pd.read_csv('schedule/compiled_schedule_suggested.csv')
    sample = pd.read_csv('schedule/compiled_sample_schedule.csv')
    perc_quality = [50, 60, 70, 80, 90]
    # %%
    suggested = suggested[suggested.algorithm == algorithm]
    # %%
    sample = sample[
        ['dimension_1', 'dimension_2', 'imp_1', 'imp_2', 'imp_3', 'od_1', 'od_2', 'imp_col_1', 'imp_col_2', 'imp_col_3',
         'quality', algorithm + '_dirty', algorithm + '_1', algorithm + '_2']]
    # %%
    original_perf = {
        'DecisionTree': 0.8489316598109451,
        'LogisticRegression': 0.7632625286934418,
        'KNN': 0.8596031818088326,
        'RandomForest': 0.8677511827887843,
        'AdaBoost': 0.8527756267575977,
        'SVC': 0.6896744361694442
    }

    # %%
    suggested_completeness = suggested[suggested.dimension_1 == 'completeness']
    #suggested_accuracy = suggested[suggested.dimension_1 == 'accuracy']
    sample_completeness = sample[sample.dimension_1 == 'completeness']
    sample_accuracy = sample[sample.dimension_1 == 'accuracy']

    # %%
    def get_median(df, col):
        median = []
        for q in perc_quality:
            median.append(np.median(df[df.quality == q][col]))
        return median

    temp_suggested = pd.DataFrame(suggested_completeness[['quality', 'perf_1', 'perf_2']])
    temp_suggested.columns = ['quality', algorithm+'_1', algorithm+'_2']
    completeness_first = pd.concat([temp_suggested, sample_completeness[['quality', algorithm+'_1', algorithm+'_2']]])

    ### confronto tra accuracy e completeness nel ranking
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_2']], algorithm+'_2'), label='accuracy first', color='royalblue')
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty', color='darkorange')
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_2'], label='accuracy first', s=10, color='royalblue')
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_dirty'], label='dirty', s=10, color='darkorange')
    plt.plot(perc_quality, get_median(completeness_first[['quality', algorithm+'_2']], algorithm+'_2'), label='completeness first', color='mediumseagreen')
    plt.scatter(completeness_first.quality, completeness_first[algorithm+'_2'], label='completeness first', s=10, color='mediumseagreen')
    plt.title(algorithm+" - ranking comparison")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    plt.ylim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i)+'_'+algorithm + ".png", bbox_inches='tight')
    i += 1
    plt.show()

    ### confronto performance del sample suggested e di quello generale (cosa succede se miglioro nella combinazione di tecniche suggerite vs miglioro tutte le combinazioni)
    plt.plot(perc_quality, get_median(suggested_completeness[['quality', 'perf_2']], 'perf_2'), label='suggested', color='red')
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty',
             color='darkorange')
    plt.scatter(suggested_completeness.quality, suggested_completeness.perf_2, label='suggested', s=10, color='red')
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_dirty'], label='dirty', s=10, color='darkorange')
    plt.title(algorithm+" - dirty vs suggested sequence (COMPLETENESS >> ACCURACY)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    plt.ylim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i)+'_'+algorithm + ".png", bbox_inches='tight')
    i += 1
    plt.show()

    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_2']], algorithm+'_2'), label='accuracy first', color='royalblue')
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty', color='darkorange')
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_2'], label='accuracy first', s=10, color='royalblue')
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_dirty'], label='dirty', s=10, color='darkorange')
    plt.plot(perc_quality, get_median(suggested_completeness[['quality', 'perf_2']], 'perf_2'), label='suggested', color='red')
    plt.scatter(suggested_completeness.quality, suggested_completeness.perf_2, label='suggested', s=10, color='red')
    plt.title(algorithm+" - dirty vs improvement (all schedules COMPLETENESS >> ACCURACY)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    plt.ylim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".png", bbox_inches='tight')
    i += 1
    plt.show()

    ### analisi di cosa succede step by step (p_dirty, p1, p2) per accuracy e completeness separatamente
    ### completeness --> accuracy
    plt.plot(perc_quality, get_median(completeness_first[['quality', algorithm+'_2']], algorithm+'_2'), label='accuracy', color='royalblue')
    plt.plot(perc_quality, get_median(completeness_first[['quality', algorithm+'_1']], algorithm+'_1'), label='completeness', color='mediumseagreen')
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty', color='darkorange')
    plt.scatter(completeness_first.quality, completeness_first[algorithm+'_2'], label='accuracy', s=10, color='royalblue')
    plt.scatter(completeness_first.quality, completeness_first[algorithm+'_1'], label='completeness', s=10, color='mediumseagreen')
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_dirty'], label='dirty', s=10, color='darkorange')
    plt.title(algorithm+" - dirty vs first vs second (COMPLETENESS >> ACCURACY)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    plt.ylim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".png", bbox_inches='tight')
    i += 1
    plt.show()

    ### accuracy --> completeness
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_1']], algorithm+'_1'), label='accuracy', color='royalblue')
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_2']], algorithm+'_2'), label='completeness', color='mediumseagreen')
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty', color='darkorange')
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_1'], label='accuracy', s=10, color='royalblue')
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_2'], label='completeness', s=10, color='mediumseagreen')
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_dirty'], label='dirty', s=10, color='darkorange')
    plt.title(algorithm+" - dirty vs first vs second (ACCURACY >> COMPLETENESS)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    plt.ylim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".png", bbox_inches='tight')
    i += 1
    plt.show()

    ### analisi di cosa succede step by step (p_dirty, p1, p2) per accuracy e completeness separatamente SOLO PER IL NOSTRO SUGGESTED SCHEDULE
    ### completeness --> accuracy
    plt.plot(perc_quality, get_median(suggested_completeness[['quality', 'perf_2']], 'perf_2'), label='accuracy', color='royalblue')
    plt.plot(perc_quality, get_median(suggested_completeness[['quality', 'perf_1']], 'perf_1'), label='completeness', color='mediumseagreen')
    plt.plot(perc_quality, get_median(suggested_completeness[['quality', 'perf_dirty']], 'perf_dirty'), label='dirty', color='darkorange')
    plt.scatter(suggested_completeness.quality, suggested_completeness.perf_2, label='accuracy', s=10, color='royalblue')
    plt.scatter(suggested_completeness.quality, suggested_completeness.perf_1, label='completeness', s=10, color='mediumseagreen')
    plt.scatter(suggested_completeness.quality, suggested_completeness.perf_dirty, label='dirty', s=10, color='darkorange')
    plt.title(algorithm+" - dirty vs first vs second (COMPLETENESS >> ACCURACY) only suggested")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    plt.ylim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".png", bbox_inches='tight')
    i += 1
    plt.show()
