import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #algorithm = 'KNN'
    #lim = [0.6, 0.85]

    #algorithm = 'LogisticRegression'
    #lim = [0.2, 0.8]

    #algorithm = 'RandomForest'
    #lim = [0.6,0.85]

    algorithm = 'SVC'
    lim = [0.1, 0.75]

    i = 0

    # %%
    ### analisi x algoritmo
    # %%

    suggested = pd.read_csv('../results_sample/compiled_schedule_suggested.csv')
    sample = pd.read_csv('../results_sample/compiled_sample_schedule.csv')
    sample = sample[sample.quality > 69]
    perc_quality = [70, 75, 80, 85, 90, 95]
    # %%
    suggested = suggested[suggested.algorithm == algorithm]
    # %%
    sample = sample[['dimension_1', 'dimension_2', 'imp_1', 'imp_2', 'imp_3', 'imp_col_1',
                     'imp_col_2', 'imp_col_3', 'od_1', 'od_2', 'od_3', 'od_imp_1',
                     'od_imp_2', 'od_imp_3', 'od_imp_col_1', 'od_imp_col_2', 'od_imp_col_3',
                     'quality', algorithm+'_dirty', algorithm+'_1', algorithm+'_2']]
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
    suggested_accuracy = suggested[suggested.dimension_1 == 'accuracy']
    sample_completeness = sample[sample.dimension_1 == 'completeness']
    sample_accuracy = sample[sample.dimension_1 == 'accuracy']


    # %%
    def get_median(df, col):
        median = []
        for q in perc_quality:
            median.append(np.median(df[df.quality == q][col]))
        return median


    temp_suggested = pd.DataFrame(suggested_accuracy[['quality', 'perf_1', 'perf_2']])
    temp_suggested.columns = ['quality', algorithm+'_1', algorithm+'_2']
    accuracy_first = pd.concat([temp_suggested, sample_accuracy[['quality', algorithm+'_1', algorithm+'_2']]])

    ### confronto tra accuracy e completeness nel ranking
    plt.plot(perc_quality, get_median(accuracy_first[['quality', algorithm+'_2']], algorithm+'_2'), label='accuracy first')
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty')
    plt.scatter(accuracy_first.quality, accuracy_first[algorithm+'_2'], label='accuracy first', s=10)
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_dirty'], label='dirty', s=10)
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_2']], algorithm+'_2'), label='completeness first')
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_2'], label='completeness first', s=10)
    plt.title(algorithm+" - ranking comparison")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    # plt.ylim([0.85,1])
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i)+'_'+algorithm + ".pdf", bbox_inches='tight')
    i += 1
    plt.show()

    ### confronto performance del sample suggested e di quello generale (cosa succede se miglioro nella combinazione di tecniche suggerite vs miglioro tutte le combinazioni)
    plt.plot(perc_quality, get_median(suggested_accuracy[['quality', 'perf_2']], 'perf_2'), label='suggested', color='red')
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty',
             color='orange')
    plt.scatter(suggested_accuracy.quality, suggested_accuracy.perf_2, label='suggested', s=10, color='red')
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_dirty'], label='dirty', s=10, color='orange')
    plt.title(algorithm+" - dirty vs suggested sequence (ACCURACY >> COMPLETENESS)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    # plt.ylim([0.85,1])
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i)+'_'+algorithm + ".pdf", bbox_inches='tight')
    i += 1
    plt.show()

    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_2']], algorithm+'_2'), label='accuracy first')
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty')
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_2'], label='accuracy first', s=10)
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_dirty'], label='dirty', s=10)
    plt.plot(perc_quality, get_median(suggested_accuracy[['quality', 'perf_2']], 'perf_2'), label='suggested', color='red')
    plt.scatter(suggested_accuracy.quality, suggested_accuracy.perf_2, label='suggested', s=10, color='red')
    plt.title(algorithm+" - dirty vs improvement (all schedules ACCURACY >> COMPLETENESS)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    # plt.ylim([0.85,1])
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1
    plt.show()

    ### analisi di cosa succede step by step (p_dirty, p1, p2) per accuracy e completeness separatamente
    ### accuracy --> completeness
    plt.plot(perc_quality, get_median(accuracy_first[['quality', algorithm+'_1']], algorithm+'_1'), label='accuracy')
    plt.plot(perc_quality, get_median(accuracy_first[['quality', algorithm+'_2']], algorithm+'_2'), label='completeness')
    plt.plot(perc_quality, get_median(sample_accuracy[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty')
    plt.scatter(accuracy_first.quality, accuracy_first[algorithm+'_1'], label='accuracy', s=10)
    plt.scatter(accuracy_first.quality, accuracy_first[algorithm+'_2'], label='completeness', s=10)
    plt.scatter(sample_accuracy.quality, sample_accuracy[algorithm+'_dirty'], label='dirty', s=10)
    plt.title(algorithm+" - dirty vs first vs second (ACCURACY >> COMPLETENESS)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    # plt.ylim([0.85,1])
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1
    plt.show()

    ### completeness --> accuracy
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_2']], algorithm+'_2'), label='accuracy')
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_1']], algorithm+'_1'), label='completeness')
    plt.plot(perc_quality, get_median(sample_completeness[['quality', algorithm+'_dirty']], algorithm+'_dirty'), label='dirty')
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_2'], label='accuracy', s=10)
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_1'], label='completeness', s=10)
    plt.scatter(sample_completeness.quality, sample_completeness[algorithm+'_dirty'], label='dirty', s=10)
    plt.title(algorithm+" - dirty vs first vs second (COMPLETENESS >> ACCURACY)")
    plt.xlabel("Quality")
    plt.ylabel("F1")
    plt.legend(bbox_to_anchor=(0.4, -0.2))
    # plt.ylim([0.85,1])
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1
    plt.show()

    ### distribution plots
    import seaborn as sns
    suggested = pd.read_csv('../results_sample/compiled_schedule_suggested.csv')
    sample = pd.read_csv('../results_sample/compiled_sample_schedule.csv')
    sample = sample[sample.quality > 69]
    perc_quality = [70, 75, 80, 85, 90, 95]
    suggested = suggested[suggested.algorithm == algorithm]

    temp_suggested = pd.DataFrame(suggested_accuracy[['dimension_1', 'quality', 'perf_dirty', 'perf_1', 'perf_2']])
    temp_suggested.columns = ['dimension_1', 'quality', algorithm+'_dirty', algorithm+'_1', algorithm+'_2']
    temp_suggested['suggested'] = 'YES'
    sample['suggested'] = 'NO'
    all = pd.concat([temp_suggested, sample[['dimension_1', 'quality', algorithm+'_dirty', algorithm+'_1', algorithm+'_2', 'suggested']]])

    ### distribuzione rankings x percentuale di qualità
    sns.displot(all, x=algorithm+"_1", hue="dimension_1", kind="kde", fill=True)
    plt.title(algorithm+" - completeness first vs accuracy first")
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    ### distribuzione rankings x percentuale di qualità
    sns.displot(all, x=algorithm+"_2", hue="dimension_1", kind="kde", fill=True)
    plt.title(algorithm+" - (ACCURACY >> COMPLETENESS) vs (COMPLETENESS >> ACCURACY)")
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    ### differenza distribuzioni tra suggested e tutte le altre combinaizoni
    sns.set_theme(style="ticks", palette="deep")
    sns.displot(all[all.suggested == 'NO'], x=algorithm+"_1", kde=True)
    plt.title(algorithm+' all schedules first dimension')
    plt.xlim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    sns.displot(all[all.suggested == 'YES'], x=algorithm+"_1", kde=True)
    plt.xlim(lim)
    plt.title(algorithm+' suggested schedule first dimension')
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    ### differenza distribuzioni tra suggested e tutte le altre combinaizoni
    sns.set_theme(style="ticks", palette="deep")
    sns.displot(all[all.suggested == 'NO'], x=algorithm+"_2", kde=True)
    plt.title(algorithm+' all schedules')
    plt.xlim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    sns.displot(all[all.suggested == 'YES'], x=algorithm+"_2", kde=True)
    plt.xlim(lim)
    plt.title(algorithm+' suggested schedule')
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    ### stessa cosa solo per accuracy --> completeness
    ### differenza distribuzioni tra suggested e tutte le altre combinaizoni

    all_accuracy = all[all.dimension_1 == 'accuracy']

    sns.set_theme(style="ticks", palette="deep")
    sns.displot(all_accuracy[all_accuracy.suggested == 'NO'], x=algorithm+"_1", kde=True)
    plt.title(algorithm+' all schedules first dimension (only accuracy -> completeness)')
    plt.xlim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    sns.displot(all_accuracy[all_accuracy.suggested == 'YES'], x=algorithm+"_1", kde=True)
    plt.xlim(lim)
    plt.title(algorithm+' suggested schedule first dimension (only accuracy -> completeness)')
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    ### differenza distribuzioni tra suggested e tutte le altre combinaizoni
    sns.set_theme(style="ticks", palette="deep")
    sns.displot(all_accuracy[all_accuracy.suggested == 'NO'], x=algorithm+"_2", kde=True)
    plt.title(algorithm+' all schedules (only accuracy -> completeness)')
    plt.xlim(lim)
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1

    sns.displot(all_accuracy[all_accuracy.suggested == 'YES'], x=algorithm+"_2", kde=True)
    plt.xlim(lim)
    plt.title(algorithm+' suggested schedule (only accuracy -> completeness)')
    plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
    i += 1


    ### per singola percentuale di qualità -- distribuzioni
    ### distribuzione rankings x percentuale di qualità

    for q in perc_quality:
        all_quality = all[all.quality == q].copy()

        ### distribuzione rankings x percentuale di qualità
        sns.displot(all_quality, x=algorithm+"_1", hue="dimension_1", kind="kde", fill=True)
        plt.title(algorithm+" - completeness first vs accuracy first " + str(q))
        plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
        i += 1

        ### distribuzione rankings x percentuale di qualità
        sns.displot(all_quality, x=algorithm+"_2", hue="dimension_1", kind="kde", fill=True)
        plt.title(algorithm+" - (ACCURACY >> COMPLETENESS) vs (COMPLETENESS >> ACCURACY) " + str(q))
        plt.savefig("/Users/camillasancricca/Desktop/" + str(i) + '_' + algorithm + ".pdf", bbox_inches='tight')
        i += 1
