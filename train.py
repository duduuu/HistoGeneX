import pandas as pd
import numpy as np
import os
import random
import re
import config as CFG
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from dataset import MAIDataset
from models import BaseModel
import utils

import warnings
warnings.filterwarnings(action='ignore') 

import importlib
importlib.reload(CFG)

df = pd.read_csv('./data/train.csv')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed) # Seed 고정

heg_genes = ['SDF4', 'NADK', 'GNB1', 'FAAP20', 'PLOD1', 'SDHB', 'AKR7A3', 'AKR7A2', 'CAPZB', 'CAMK2N1', 'HSPG2', 'C1QC', 'C1QB', 'ATP5IF1', 'SFPQ', 'RPS8', 'PRDX1', 'CMPK1', 'PGM1', 'SRSF11', 'CNN3', 'AGL', 'ATP5PB', 'ATP1A1', 'MRPS21', 'S100A10', 'S100A8', 'S100A6', 'S100A13', 'RAB13', 'RPS27', 'TAGLN2', 'PEX19', 'TMCO1', 'PRDX6', 'GLUL', 'CFHR1', 'SRP9', 'RNF187', 'RAB4A', 'EGLN1', 'TOMM20', 'NID1', 'RHOB', 'APOB', 'GCKR', 'PPP1CB', 'ACTR2', 'PCBP1', 'SNRPG', 'TMSB10', 'GGCX', 'PLGLB2', 'FABP1', 'IGKC', 'FAHD2A', 'SNRNP200', 'RPL31', 'PROC', 'MZT2B', 'TFPI', 'RPL37A', 'IGFBP5', 'RAB17', 'NDUFA10', 'RPL15', 'RPSA', 'PTH1R', 'RPL29', 'STAB1', 'SPCS1', 'NIT2', 'ALDH1L1', 'CHST13', 'ACAD11', 'DNAJC19', 'ST6GAL1', 'SPON2', 'SH3BP2', 'LRPAP1', 'RPL9', 'SMIM14', 'OCIAD2', 'IGFBP7', 'ADH6', 'OSTC', 'CMBL', 'MARCH6', 'GOLPH3', 'SUB1', 'AMACR', 'C9', 'C7', 'CCDC152', 'HEXB', 'IQGAP2', 'BHMT', 'COX7C', 'REEP5', 'DIAPH1', 'CD74', 'RPS14', 'GPX3', 'SPARC', 'FAXDC2', 'HIGD2A', 'TMED9', 'CANX', 'MGAT1', 'ACOT13', 'ZSCAN16-AS1', 'HLA-A', 'RNF5', 'TOMM6', 'CNPY3', 'SLC22A7', 'ELOVL5', 'SMLR1', 'RPS12', 'PERP', 'HEBP2', 'EZR', 'EIF3B', 'INMT', 'IGFBP1', 'DDC', 'GUSB', 'HSPB1', 'CYP51A1', 'CYP3A4', 'LAMTOR4', 'TFR2', 'PMPCB', 'NDUFA5', 'CALD1', 'PDIA4', 'EIF4EBP1', 'CEBPD', 'RPS20', 'RAB2A', 'RPL7', 'RIDA', 'PABPC1', 'ENY2', 'NDRG1', 'LY6E', 'NRBP2', 'PLEC', 'GRINA', 'GPAA1', 'HSF1', 'GPT', 'RPL8', 'MPDZ', 'ALDH1B1', 'HDHD3', 'ORM1', 'ENG', 'ST6GALNAC6', 'FBXW5', 'AKR1C4', 'GDI2', 'FAM107B', 'VIM', 'MSRB2', 'ITGB1', 'CCNY', 'SAR1A', 'PPA1', 'ANAPC16', 'CHCHD1', 'PANK1', 'PGAM1', 'SCD', 'MGMT', 'PRAP1', 'ECHS1', 'CYP2E1', 'IFITM3', 'RNH1', 'RPLP2', 'CHID1', 'H19', 'CD81', 'HBB', 'SAA2', 'SLC43A3', 'FADS1', 'SF1', 'SYVN1', 'NEAT1', 'MALAT1', 'CFL1', 'SF3B2', 'DHCR7', 'NUMA1', 'COA4', 'SLCO2B1', 'RPS3', 'TSKU', 'NDUFC2', 'APOA5', 'HSPA8', 'APLP2', 'MLF2', 'SLCO1B3', 'SLC38A2', 'SLC38A4', 'ERBB3', 'ATP5F1B', 'DCN', 'HAL', 'NR1H4', 'ACACB', 'RPL6', 'COX6A1', 'DYNLL1', 'TMED2', 'SCARB1', 'POMP', 'ZC3H13', 'F7', 'PROZ', 'RNASE1', 'NFKBIA', 'SLC10A1', 'TMED10', 'GSTZ1', 'SLIRP', 'IGHA1', 'IGHM', 'EMC4', 'SERF2', 'GATM', 'SPPL2A', 'ANXA2', 'HEXA', 'CYP1A2', 'COX5A', 'COMMD4', 'MORF4L1', 'RPS17', 'ANPEP', 'HBA2', 'HBA1', 'GNPTG', 'NME3', 'TRAP1', 'ABAT', 'CARHSP1', 'EMP2', 'PDXDC1', 'ARL6IP1', 'NUPR1', 'MT1F', 'HERPUD1', 'CES2', 'ATP6V0D1', 'LDHD', 'APRT', 'SPG7', 'CCL14', 'PSMD3', 'CNP', 'NBR1', 'GRN', 'ATP5MC1', 'PDK2', 'PCTP', 'RPL38', 'GPRC5C', 'SLC38A10', 'ACTG1', 'P4HB', 'RFNG', 'RPL17', 'MED16', 'OAZ1', 'LSM7', 'TIMM13', 'THOP1', 'TLE5', 'MAP2K2', 'ALKBH7', 'ZNF358', 'RAB11B', 'SHFL', 'ANGPTL8', 'C19orf53', 'TECR', 'CYP4F11', 'AP1M1', 'RPL18A', 'UBA52', 'CEBPA', 'USF2', 'ACTN4', 'ZFP36', 'CYP2B6', 'RPS19', 'CALM3', 'LINC01595', 'RPL28', 'ISOC2', 'DDRGK1', 'RRBP1', 'CST3', 'ROMO1', 'SCAND1', 'NORAD', 'MYL9', 'ATP5F1E', 'OGFR', 'ETS2', 'COL18A1', 'LSS', 'SLC25A1', 'TIMP3', 'PMM1', 'CYP2D6', 'TSPO', 'SELENOO', 'PPP6R2', 'LMF2', 'CD99', 'TMSB4X', 'SYAP1', 'PRDX4', 'SAT1', 'UBA1', 'RBM3', 'AR', 'COX7B', 'SLC25A5', 'RPL39', 'MT-ND1', 'MT-ND2', 'MT-CO1', 'MT-CO2', 'MT-ATP8', 'MT-ATP6', 'MT-CO3', 'MT-ND3', 'MT-ND4L', 'MT-ND4', 'MT-ND5', 'MT-CYB']
hvg_genes = ['TNFRSF18', 'LINC01770', 'C1orf167', 'CTRC', 'HSPB7', 'FAM43B', 'AC004865.2', 'AL139286.2', 'AL033527.3', 'ARTN', 'AL136985.2', 'AC093424.1', 'DNAJC6', 'AL031429.2', 'LINC01781', 'LINC02795', 'LRRC8C-DT', 'VTCN1', 'REG4', 'AC245014.1', 'PPIAL4G', 'HIST2H2AB', 'KCNN3', 'LINC01704', 'CADM3', 'SH2D1B', 'RGS13', 'CRB1', 'GPR37L1', 'LRRN2', 'CNTN2', 'C1orf116', 'SERTAD4-AS1', 'SPATA17', 'STUM', 'NLRP3', 'AC104695.4', 'AC006369.1', 'EPCAM-DT', 'BCL11A', 'AC007878.1', 'AC092653.1', 'IGKV1-27', 'NCAPH', 'GCC2-AS1', 'AC092645.2', 'CFAP221', 'GALNT13', 'SCN3A', 'LINC01934', 'HECW2-AS1', 'KIAA2012', 'AC064836.2', 'AC007383.3', 'AC007879.3', 'IQCA1', 'RBM44', 'CAV3', 'SUSD5', 'LINC02158', 'ZNF660', 'KIF15', 'TMIE', 'TMEM89', 'TNNC1', 'DNAH12', 'TAFA1', 'FOXP1-IT1', 'GPR27', 'EBLN2', 'AC107204.1', 'NECTIN3-AS1', 'AC112484.1', 'AC107027.3', 'CPA3', 'PTX3', 'NLGN1', 'SOX2', 'YEATS2-AS1', 'AC046143.2', 'AC117490.2', 'AC024560.5', 'JAKMIP1', 'AC006230.1', 'LGI2', 'AC069307.1', 'ADGRL3', 'PRDM8', 'AC108021.1', 'PRDM5', 'ADAD1', 'AC053545.1', 'AC026402.2', 'PABPC4L', 'MGAT4D', 'ZNF827', 'MAB21L2', 'AC093599.2', 'C4orf47', 'FRG1-DT', 'LINC02160', 'AC025171.3', 'AC025171.2', 'LRRC70', 'AC010359.3', 'SLC22A4', 'AC104116.1', 'AC011379.1', 'CAMK2A', 'AL357054.4', 'AL031123.1', 'ERVFRD-1', 'AL022098.1', 'CASC15', 'RNF39', 'SFTA2', 'HLA-DOB', 'TMEM217', 'AL355802.3', 'AL096865.1', 'BEND6', 'AL445250.1', 'AL391840.3', 'AL139274.2', 'GABRR2', 'AL390208.1', 'TRAPPC3L', 'THEMIS', 'MTFR2', 'IL20RA', 'AL355312.3', 'AL080276.2', 'LINC02538', 'AC010991.1', 'AC005014.3', 'ABCB5', 'HOXA10', 'AC007255.1', 'LINC02848', 'SEMA3D', 'AC004522.3', 'AC254629.1', 'AC105446.1', 'SLC26A4', 'LEP', 'AC245519.1', 'AF131216.3', 'AC108449.3', 'UNC5D', 'AC090152.1', 'STMN2', 'LINC02849', 'HAS2', 'AC016405.3', 'FAM83A', 'CCDC26', 'AC100803.4', 'AC084125.2', 'VLDLR-AS1', 'ANKRD20A1', 'C9orf153', 'AL360020.1', 'AL449403.2', 'AL512590.3', 'PTCSC2', 'AL807776.1', 'FAM225B', 'TNFSF15', 'AC007066.3', 'LCN8', 'LINC00702', 'CALML3-AS1', 'AL137186.1', 'AL158211.1', 'AC010864.1', 'AL450326.1', 'AL133551.1', 'TACR2', 'AL731563.2', 'MYOZ1', 'ACTA2-AS1', 'GOLGA7B', 'C10orf95', 'AL731571.1', 'FOXI2', 'EBF3', 'SYT8', 'TNNT3', 'KCNQ1-AS1', 'STK33', 'AC090559.2', 'GLYATL2', 'AP001636.3', 'SLC22A11', 'NPAS4', 'CNTN5', 'AP000866.2', 'ADAMTS8', 'SLCO1C1', 'CCDC65', 'KCNH3', 'ASIC1', 'AC008147.2', 'AC025259.3', 'AC026124.2', 'MYRFL', 'AC078923.1', 'AC025034.1', 'RPH3A', 'TMEM132D', 'AC148476.1', 'PDX1', 'MEDAG', 'MYCBP2-AS1', 'AC001226.1', 'EDNRB-AS1', 'AL137781.1', 'SLITRK5', 'AL133371.2', 'LINC02332', 'CMTM5', 'AL136018.1', 'AL049830.3', 'AL365295.1', 'AL157911.1', 'SYNDIG1L', 'AL049780.1', 'ESRRB', 'ASB2', 'LINC01550', 'LINC00605', 'AC022613.2', 'AC023908.3', 'PHGR1', 'SERINC4', 'AC090527.3', 'AC068722.1', 'MYEF2', 'LINC01169', 'NOX5', 'AC100827.4', 'ADPGK-AS1', 'LINC02253', 'FAM169B', 'AC090825.1', 'AL023881.1', 'MSLN', 'AC120498.9', 'PRSS21', 'PRSS22', 'BICDL2', 'AC108134.2', 'AC006111.2', 'SMIM22', 'LINC02858', 'AC009034.1', 'SCNN1B', 'IL21R', 'CD19', 'AC120114.2', 'TLCD3B', 'NPIPB13', 'AC009088.1', 'AC007906.2', 'AC092140.2', 'AC027682.4', 'AC010530.1', 'CHST6', 'CHST5', 'AC105429.1', 'USP6', 'AC005410.2', 'AC005838.2', 'LINC02094', 'LINC01563', 'CCL13', 'CCR7', 'AC100793.4', 'ETV4', 'AC003070.2', 'EPN3', 'AC025048.4', 'CD300E', 'AC021683.3', 'AC021683.2', 'AC021683.5', 'LINC01979', 'AC145207.6', 'AP005530.1', 'AP001020.3', 'DLGAP1', 'AC006566.1', 'LINC01444', 'AQP4-AS1', 'PMAIP1', 'AC090409.1', 'AC125437.1', 'LINC01002', 'CATSPERD', 'TNFSF9', 'TEX45', 'PODNL1', 'PBX4', 'ZNF66', 'ZNF208', 'LINC01801', 'CD22', 'GAPDHS', 'APLP1', 'ZNF793-AS1', 'LINC01480', 'CGB7', 'LILRA4', 'ANGPT4', 'SIRPG', 'ADRA1D', 'GGTLC1', 'ZNF341-AS1', 'AL121895.2', 'AL133342.1', 'DOK5', 'ZBP1', 'APCDD1L', 'C20orf197', 'MIR1-1HG-AS1', 'RBM11', 'MIS18A-AS1', 'B3GALT5', 'ERVH48-1', 'AP001053.1', 'MCM3AP-AS1', 'AP001469.3', 'AP000553.2', 'IGLV6-57', 'IGLC6', 'DRICH1', 'AL022476.1', 'LINC01639', 'AL031595.3', 'KLHDC7B', 'AL590764.1', 'AL022151.1', 'Z68871.1', 'ZCCHC18', 'PAK3', 'MCF2', 'AC234781.1']

def train(model, fold, optimizer, train_loader, val_loader, scheduler, device):
    criterion = nn.MSELoss().to(device)
    best_loss = float('inf')
        
    for epoch in tqdm(range(CFG.epochs)):
        # train
        model.train()
        train_loss = []
        for imgs, labels in iter(train_loader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            pred = model(imgs)
            
            if CFG.loss_function == "pearson":
                loss = utils.negative_pearson_loss(pred, labels)
            else:
                loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        # validation
        model.eval()
        val_loss = []
        preds = []
        with torch.no_grad():
            for imgs, labels in iter(val_loader):
                imgs = imgs.float().to(device)
                labels = labels.to(device)
                
                pred = model(imgs)
                
                if CFG.loss_function == "pearson":
                    loss = utils.negative_pearson_loss(pred, labels)
                else:
                    loss = criterion(pred, labels)
                    
                val_loss.append(loss.item())
                
                preds.append(pred.detach().cpu())
        
        preds = torch.cat(preds).numpy()
        meancorr_cells = np.mean([pearsonr(preds[i], val_label_vec[i])[0] for i in range(preds.shape[0])])
        maxcorr_genes = np.mean([pearsonr(preds[:, j], val_label_vec[:, j])[0] for j in range(preds.shape[1])])
        
        _train_loss = np.mean(train_loss)
        _val_loss = np.mean(val_loss)
        
        wandb.log({
            f"fold_{fold+1}_train_loss": _train_loss, 
            f"fold_{fold+1}_val_loss": _val_loss, 
            f"fold_{fold+1}_meancorr_cells": meancorr_cells, 
            f"fold_{fold+1}_maxcorr_genes": maxcorr_genes
        }, step=epoch)
       
        if scheduler is not None:
            scheduler.step(_val_loss)
            
        if best_loss > _val_loss:
            best_loss = _val_loss
            best_model = model
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > CFG.early_stop_count:
                break
            
    return best_model, best_loss, best_epoch

NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)

wandb_name = CFG.model_name
if CFG.augment == True:
    wandb_name += "_aug"

wandb.init(project='MAI', name=wandb_name, group=CFG.model_name)

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"Fold {fold+1}")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_label_vec = train_df.iloc[:,2:].values.astype(np.float32)
    val_label_vec = val_df.iloc[:,2:].values.astype(np.float32)

    train_dataset = MAIDataset(train_df['path'].values, train_label_vec, augment=CFG.augment)
    train_loader = DataLoader(train_dataset, batch_size = CFG.batch_size, shuffle=True, num_workers=0)    

    val_dataset = MAIDataset(val_df['path'].values, val_label_vec)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)
    
    model = BaseModel(CFG.model_name)
    model.to(device)
        
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG.learning_rate)
    
    if CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=CFG.min_lr)
    else:
        raise ValueError(f"Unknown scheduler name: {CFG.scheduler}.")
    
    best_model, best_loss, best_epoch = train(model, fold, optimizer, train_loader, val_loader, scheduler, device)
    
    model_save_path = f"./models/{wandb_name}_fold{fold+1}_epoch{best_epoch+1}_{best_loss:.5f}.pth"
    torch.save(best_model.state_dict(), model_save_path)

wandb.finish()