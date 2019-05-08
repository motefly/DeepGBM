import pandas as pd
import numpy as np
import category_encoders as ce
from tqdm import tqdm
import argparse, collections, os
import gc
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('-r', '--thresrate', type=float, default=0.99)
parser.add_argument('-i', '--numI', type=int, default=int(13))
parser.add_argument('-c', '--numC', type=int, default=int(26))

parser.add_argument('--train_csv_path', type=str)
parser.add_argument('--test_csv_path', type=str)
parser.add_argument('out_dir', type=str)

parser.add_argument('--online', action='store_true')
parser.add_argument('--data', type=str)
parser.add_argument('--num_onlines', type=int)

args = vars(parser.parse_args())

def unpackbits(x,num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

class NumEncoder(object):
    def __init__(self, cate_col, nume_col, threshold, thresrate, label):
        self.label_name = label
        # cate_col = list(df.select_dtypes(include=['object']))
        self.cate_col = cate_col
        # nume_col = list(set(list(df)) - set(cate_col))
        self.dtype_dict = {}
        for item in cate_col:
            self.dtype_dict[item] = 'str'
        for item in nume_col:
            self.dtype_dict[item] = 'float'
        self.nume_col = nume_col
        self.tgt_nume_col = []
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col)
        self.threshold = threshold
        self.thresrate = thresrate
        # for online update, to do
        self.save_cate_avgs = {}
        self.save_value_filter = {}
        self.save_num_embs = {}
        self.Max_len = {}
        self.samples = 0

    def fit_transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Fitting and Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        self.samples = df.shape[0]
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            self.save_value_filter[item] = list(value_counts[:int(num*self.thresrate)][value_counts>self.threshold].index)
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')
            del value_counts
            gc.collect()

        for item in tqdm(self.nume_col):
            df[item] = df[item].fillna(df[item].mean())
            self.save_num_embs[item] = {'sum':df[item].sum(), 'cnt':df[item].shape[0]}

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.fit_transform(df)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_col):
            feats = df[item].values
            labels = df[self.label_name].values
            feat_encoding = {'mean':[], 'count':[]}
            feat_temp_result = collections.defaultdict(lambda : [0, 0])
            self.save_cate_avgs[item] = collections.defaultdict(lambda : [0, 0])
            for idx in range(self.samples):
                cur_feat = feats[idx]
                # smoothing optional
                if cur_feat in self.save_cate_avgs[item]:
                    # feat_temp_result[cur_feat][0] = 0.9*feat_temp_result[cur_feat][0] + 0.1*self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1]
                    # feat_temp_result[cur_feat][1] = 0.9*feat_temp_result[cur_feat][1] + 0.1*self.save_cate_avgs[item][cur_feat][1]/idx
                    feat_encoding['mean'].append(self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1])
                    feat_encoding['count'].append(self.save_cate_avgs[item][cur_feat][1]/idx)
                else:
                    feat_encoding['mean'].append(0)
                    feat_encoding['count'].append(0)
                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item+'_t_mean'] = feat_encoding['mean']
            df[item+'_t_count'] = feat_encoding['count']
            self.tgt_nume_col.append(item+'_t_mean')
            self.tgt_nume_col.append(item+'_t_count')
        
        print('Start manual binary encode')
        rows = None
        for item in tqdm(self.nume_col+self.tgt_nume_col):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_col):
            feats = df[item].values
            Max = df[item].max()
            bit_len = len(bin(Max)) - 2
            samples = self.samples
            self.Max_len[item] = bit_len
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        trn_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        trn_x = np.array(rows)
        np.save(outPath+'_features.npy', trn_x)
        np.save(outPath+'_labels.npy', trn_y)

    # for test dataset
    def transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        samples = df.shape[0]
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_col):
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_col):
            avgs = self.save_cate_avgs[item]
            df[item+'_t_mean'] = df[item].map(lambda x: avgs[x][0]/avgs[x][1] if x in avgs else 0)
            df[item+'_t_count'] = df[item].map(lambda x: avgs[x][1]/self.samples if x in avgs else 0)
        
        print('Start manual binary encode')
        rows = None
        for item in tqdm(self.nume_col+self.tgt_nume_col):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_col):
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        vld_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        vld_x = np.array(rows)
        np.save(outPath+'_features.npy', vld_x)
        np.save(outPath+'_labels.npy', vld_y)
    
    # for update online dataset
    def refit_transform(self, inPath, outPath):
        print('----------------------------------------------------------------------')
        print('Refitting and Transforming %s .'%inPath)
        print('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        samples = df.shape[0]
        print('Filtering and fillna features')
        for item in tqdm(self.cate_col):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_col):
            self.save_num_embs[item]['sum'] += df[item].sum()
            self.save_num_embs[item]['cnt'] += df[item].shape[0]
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_col):
            feats = df[item].values
            labels = df[self.label_name].values
            feat_encoding = {'mean':[], 'count':[]}
            for idx in range(samples):
                cur_feat = feats[idx]
                if self.save_cate_avgs[item][cur_feat][1] == 0:
                    pdb.set_trace()
                feat_encoding['mean'].append(self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1])
                feat_encoding['count'].append(self.save_cate_avgs[item][cur_feat][1]/(self.samples+idx))
                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item+'_t_mean'] = feat_encoding['mean']
            df[item+'_t_count'] = feat_encoding['count']

        self.samples += samples
            
        print('Start manual binary encode')
        rows = None
        for item in tqdm(self.nume_col+self.tgt_nume_col):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_col):
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        vld_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        vld_x = np.array(rows)
        np.save(outPath+'_features.npy', vld_x)
        np.save(outPath+'_labels.npy', vld_y)
        # to do
        pass

if __name__ == '__main__':
    
    # for criteo
    # cate_col = ['C'+str(i) for i in range(1, args['numC']+1)]
    # nume_col = ['I'+str(i) for i in range(1, args['numI']+1)]
    # label_col = 'Label'

    # for flight delay
    # cate_col = ["Month_cate", "DayofMonth_cate", "DayOfWeek_cate", "DepTime_cate", "UniqueCarrier", "Origin", "Dest"]
    # nume_col = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"]
    # label_col = 'dep_delayed_15min'

    # for bike demand
    # cate_col = ['month_cate', 'day_cate', 'hour_cate', 'dayofweek_cate', 'season', 'weather_cate']
    # nume_col = ['month', 'day', 'hour', 'dayofweek', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']
    # label_col = 'count'

    # for talking
    # nume_col = ['click_hour']
    # cate_col = ['ip','app','device','os','channel','click_hour_cate']
    # label_col='is_attributed'

    # for zillow
    # nume_col = ['bathroomcnt','bedroomcnt','calculatedbathnbr','threequarterbathnbr','finishedfloor1squarefeet','calculatedfinishedsquarefeet','finishedsquarefeet6','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50','fireplacecnt','fullbathcnt','garagecarcnt','garagetotalsqft','latitude','longitude','lotsizesquarefeet','numberofstories','poolcnt','poolsizesum','roomcnt','unitcnt','yardbuildingsqft17','yardbuildingsqft17','taxvaluedollarcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount','taxdelinquencyyear','yearbuilt']
    # cate_col = ['architecturalstyletypeid', 'yearbuilt_cate', 'buildingqualitytypeid', 'propertyzoningdesc', 'regionidneighborhood', 'yardbuildingsqft26', 'fireplaceflag', 'propertycountylandusecode', 'hashottuborspa', 'basementsqft', 'fips', 'buildingclasstypeid', 'pooltypeid2', 'pooltypeid10', 'regionidcounty', 'heatingorsystemtypeid', 'rawcensustractandblock', 'censustractandblock', 'taxdelinquencyflag', 'airconditioningtypeid', 'pooltypeid7', 'regionidcity', 'regionidzip', 'decktypeid', 'typeconstructiontypeid', 'propertylandusetypeid', 'storytypeid']
    # label_col = 'logerror'

    # for malware
    # nume_col = ['AVProductsInstalled', 'AVProductsEnabled','Census_ProcessorCoreCount','Census_PrimaryDiskTotalCapacity','Census_SystemVolumeTotalCapacity','Census_TotalPhysicalRAM','Census_InternalPrimaryDiagonalDisplaySizeInInches','Census_InternalPrimaryDisplayResolutionHorizontal','Census_InternalPrimaryDisplayResolutionVertical','Census_InternalBatteryNumberOfCharges','Census_OSBuildNumber','Census_OSBuildRevision']
    # cate_col = ['IeVerIdentifier', 'Census_ProcessorClass', 'Processor', 'Census_OEMNameIdentifier', 'Firewall', 'Census_FirmwareVersionIdentifier', 'AppVersion', 'CityIdentifier', 'Census_PowerPlatformRoleName', 'Census_OSBranch', 'AvSigVersion', 'Census_IsPortableOperatingSystem', 'Census_OSEdition', 'Census_GenuineStateName', 'OsVer', 'Census_IsAlwaysOnAlwaysConnectedCapable', 'HasTpm', 'Census_IsWIMBootEnabled', 'Census_IsFlightsDisabled', 'Census_IsFlightingInternal', 'AutoSampleOptIn', 'SkuEdition', 'SMode', 'Census_OSWUAutoUpdateOptionsName', 'Wdft_IsGamer', 'Census_OSUILocaleIdentifier', 'Census_IsPenCapable', 'OsPlatformSubRelease', 'Census_IsTouchEnabled', 'IsBeta', 'Census_HasOpticalDiskDrive', 'SmartScreen', 'IsProtected', 'Census_ProcessorModelIdentifier', 'Census_PrimaryDiskTypeName', 'OrganizationIdentifier', 'Census_ActivationChannel', 'Census_IsSecureBootEnabled', 'Census_OSArchitecture', 'CountryIdentifier', 'Census_ThresholdOptIn', 'Census_ChassisTypeName', 'Census_OSSkuName', 'Census_FirmwareManufacturerIdentifier', 'PuaMode', 'Census_MDC2FormFactor', 'ProductName', 'AVProductStatesIdentifier', 'GeoNameIdentifier', 'Census_OSInstallLanguageIdentifier', 'Census_ProcessorManufacturerIdentifier', 'Census_IsVirtualDevice', 'UacLuaenable', 'Census_OSInstallTypeName', 'Platform', 'Census_DeviceFamily', 'Census_InternalBatteryType', 'RtpStateBitfield', 'DefaultBrowsersIdentifier', 'OsBuild', 'OsSuite', 'EngineVersion', 'Census_FlightRing', 'IsSxsPassiveMode', 'Census_OSVersion', 'Wdft_RegionIdentifier', 'LocaleEnglishNameIdentifier', 'Census_OEMModelIdentifier', 'OsBuildLab']
    # label_col = 'HasDetections'

    # for nips_a
    nume_col = ['13', '14', '15', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '41', '42', '43', '44', '45', '46', '47', '48', '49', '54']
    cate_col = ['0', '2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '16', '17', '18', '27', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '50', '51', '52', '53', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '72', '74', '75', '78', '80', '81']
    label_col = 'label'

    # for nips_d
    # nume_col = ['3', '5', '6', '7', '11', '12', '16', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '46', '49', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '70', '71', '72', '73', '74', '75']
    # cate_col = ['0', '1', '2', '4', '9', '10', '13', '14', '15', '17', '39', '40', '43', '44', '45', '47', '69']
    # label_col = 'label'

    # for nips b
    # nume_col = ['9','11','12','13','14','15','16']
    # cate_col = ['0','1','2','3','4','6','7','8','10','17','18','19','20','21','22','23','24']
    # label_col = 'label'

    if not os.path.isdir(args['out_dir']):
        os.mkdir(args['out_dir'])
    ec = NumEncoder(cate_col, nume_col, args['threshold'], args['thresrate'], label_col)
    def online_encoding():
        in_map = lambda x : args['data']+'_online%d.csv'%x
        out_map = lambda x : args['out_dir']+'/online%d'%x
        
        ec.fit_transform(in_map(0), out_map(0)+'_train')
        ec.transform(in_map(1), out_map(1)+'_test')
        
        for idx in range(1, args['num_onlines']-1):
            ec.refit_transform(in_map(idx), out_map(idx)+'_train')
            ec.transform(in_map(idx+1), out_map(idx+1)+'_test')

    if args['online']:
        online_encoding()
    else:
        ec.fit_transform(args['train_csv_path'], args['out_dir']+'/train')
        ec.transform(args['test_csv_path'], args['out_dir']+'/test')
