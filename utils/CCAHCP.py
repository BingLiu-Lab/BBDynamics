import numpy as np
from scipy import linalg
import statsmodels.api as sm
import pandas as pd
from sklearn import preprocessing
import os
import scipy.stats
from tqdm import tqdm
import seaborn as sns
from seaborn import heatmap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA as skCCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import nibabel as nib
import sys
sys.path.append('PATH_TO_PLOT_SURFACE_DATA')
import viz

## measures : 159 behavior phenotypes used for analysis
measures = pd.read_csv('../data/HCPYA_behavior/hcp_behavior_list.csv')['variable'].values.tolist()
measures_include_info = measures + ['Subject','Gender']

def viz_xdata(data,title,vmax = None,threshold=None, h = True, darkness = 0.6,folder = 'ROI-VIZ2',colormap ='bwr'):

    dataL,dataR = viz.Schaefer2HCP(data)
    if h :
        fig = viz.viz_4view_h(stat_map_R = dataR,stat_map_L= dataL, vmax = vmax,threshold = threshold,
                              symmetric_cbar = True, darkness = darkness,colormap = colormap,
                              bg_on_data =True ,figpath = os.path.join(os.getcwd(),folder), dpi = 80,
                              figname = title,datazscored = False)
    else:
        fig = viz.viz_4view(stat_map_R = dataR,stat_map_L= dataL,bg_on_data =True, colormap =colormap,
                            vmax = vmax,threshold = threshold,figpath = os.path.join(os.getcwd(),folder),
                            symmetric_cbar = True,dpi = 80, darkness = darkness,
                            figname = title,datazscored = False)        
    return fig

def get_feature_behavior_corr(feature_mat,beh_vec,nroi = 271,p_thres = 0.05):
    res_corr = []
    for i in range(nroi):
        r,p = scipy.stats.spearmanr(feature_mat[:,i] ,beh_vec)
        res_corr.append([r,p])
    corrdf = pd.DataFrame(np.array(res_corr),columns = ['r','p'])
    corrdf['r_sig'] = np.where(corrdf['p']<p_thres,corrdf['r'],0)
    return corrdf

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def clean_confounds(cfdf):
    ## preproc for confounds data
    cf_data = cfdf.values
    cf_data = np.nan_to_num(cf_data)
    cf_data_std = StandardScaler().fit_transform(cf_data)
    return cf_data_std

def get_residual(data_std,cf_data_std):
    ## regress data via OLS
    OLS_model = sm.OLS(data_std,cf_data_std).fit()  
    return OLS_model.resid 

def pca_input(data,n_components = 100):
    pca_model = PCA(n_components=n_components)
    valpca = pca_model.fit_transform(data)
    # print(' explained variance:', pca_model.explained_variance_ratio_.sum())
    return valpca

    
def clean_input_xy(y_df,x,behavior_list):
    ## Standard input X matrix for CCA analysis
    xstd = StandardScaler()
    xdata_std = xstd.fit_transform(x)
    
    ## Standard input Y matrix for CCA analysis
    yval = y_df[behavior_list].values
    ystd = StandardScaler()
    ydata_std = ystd.fit_transform(yval)
    return xdata_std,ydata_std


def preproc_CCA(Xinp,Yinpdf,Cfdf,behavior_list,cf_list):
    # preproc for input X and Y
    nsub,nroi,nf = Xinp.shape
    Xinp2d = Xinp.reshape(nsub,nroi*nf)
    xclean,yclean = clean_input_xy(Yinpdf,Xinp2d,behavior_list)
    
    # preproc for confounds mat
    cf_data = clean_confounds(Cfdf[cf_list])
    
    # regress confounder from X and Y 
    y_residual  = get_residual(yclean,cf_data)
    x_residual  = get_residual(xclean,cf_data)

    # PCA Dimension reduction on residual input X_residual and Y_residual 
    y_pca  = pca_input(y_residual)
    x_pca  = pca_input(x_residual)
    
    return_dict = {
             'xpca' : x_pca, 'ypca' : y_pca,
             'xclean' : xclean,'yclean' : yclean,
             'nfeature': nf,'ydf':Yinpdf
    }
    return return_dict
    
    
def power_normalization(data):
    data_dim = data.shape[0]
    return preprocessing.PowerTransformer().fit_transform( data.reshape(-1,1) ).reshape(data_dim,)

def _check_clean_dfy(df,p_thres = 0.05):
    ## loc significant correlated behavior phenotype, reorder by abs weight
    df.insert(1,'rabs',np.abs(df.r))
    return df.loc[df.p<p_thres].sort_values('rabs',ascending = False)

def resume_pattern_y(index,Y_score,yclean):
    resm = []
    for i in range(len(measures)):
        r,p = scipy.stats.spearmanr(Y_score[:,index],yclean[:,i])
        resm.append([measures[i],r,p])
    dfm = pd.DataFrame(resm,columns = ['y','r','p'])
    dfmc = _check_clean_dfy(dfm)
    res = {'org':dfm,'clean':dfmc}
    return res

def resume_pattern_x(index,X_score,xclean,nfeature,nroi = 271,p_thres = 0.05):
    xm1r = []
    xm1p = []
    for i in range(xclean.shape[1]):
        r,p = scipy.stats.pearsonr(X_score[:,index],xclean[:,i])
        xm1r.append(r)
        xm1p.append(p)
    xm1r = np.array(xm1r).reshape(nroi,nfeature)
    xm1p = np.array(xm1p).reshape(nroi,nfeature)
    xmode_corr_mask_m1 = np.where(xm1p < p_thres,xm1r,0)
    xmode_corr_mask_m1_mean = np.abs(xmode_corr_mask_m1).mean(axis =1)
    
    res = { 'org' : xmode_corr_mask_m1_mean , 
            'clean': normalization(power_normalization(xmode_corr_mask_m1_mean)),
            'mat_mask':   xmode_corr_mask_m1                 }
    return res

def resume_pattern_results(data_dict,pattern_resdict):
    
    Y_score = pattern_resdict['model'].y_scores_ ## CCA behavior score
    X_score = pattern_resdict['model'].x_scores_ ## CCA brain score
    
    ## get top significant pattern for brain feature and behavior feature
    mode_dict = {}
    for m,mindex in zip(['m1','m2'],[0,1]):
        mode_dict[m] = {}
        mode_dict[m]['x'] = resume_pattern_x(mindex,X_score,data_dict['xclean'],data_dict['nfeature'])
        mode_dict[m]['y'] = resume_pattern_y(mindex,Y_score,data_dict['yclean'])
    return mode_dict

def cca_analysis(inpdata,n_components = 10 ):
    ## input data for CCA analysis
    x_pca = inpdata['xpca']
    y_pca = inpdata['ypca']
    
    ## perform CCA via sklearn
    cca = skCCA(n_components = n_components)
    cca.fit(x_pca,y_pca)
    
    ## correlation coefficient for each mode
    s = np.corrcoef(cca.x_scores_.T, 
                    cca.y_scores_.T).diagonal(
                                            offset=cca.n_components)
    a = cca.x_weights_
    b = cca.y_weights_
    
    ## CCA brain score & CCA behavior score
    X_score, Y_score = cca.x_scores_, cca.y_scores_
    # print(s)
    result_dict = {'model':cca,'s':s,'a':a,'b':b}
    return result_dict

def load_data_into_atlas(datalist,atlas,nroi):
    ## load weights list onto atlas
    atlas_mat = np.asarray(atlas.dataobj)
    data_on_atlas = np.zeros(atlas_mat.shape)
    for i in range(nroi):
        index = np.where(atlas_mat == i+1)
        data_on_atlas[index] = datalist[i]
    data_nifti = nib.Nifti1Image(data_on_atlas.astype(float), atlas.affine)
    return data_nifti

