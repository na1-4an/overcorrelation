# ### DeProp-Cora
# python main_deprop.py --dataset=Cora --weight_decay=5e-4 --dropout=0.5 --lr=0.01 --epoch=1000 --cuda_num=0
# python main_deprop.py --dataset=Cora --weight_decay=5e-4 --dropout=0.0 --lr=0.01 --epoch=1000 --cuda_num=0

### DeProp-Citeseer
python main_deprop.py --dataset=Citeseer --weight_decay=5e-4 --dropout=0.5 --lr=0.01 --epoch=1000 --cuda_num=0
python main_deprop.py --dataset=Citeseer --weight_decay=5e-4 --dropout=0.0 --lr=0.01 --epoch=1000 --cuda_num=0

### DeProp-Pubmed
python main_deprop.py --dataset=Pubmed --weight_decay=5e-4 --dropout=0.5 --lr=0.01 --epoch=1000 --cuda_num=0
python main_deprop.py --dataset=Pubmed --weight_decay=5e-4 --dropout=0.0 --lr=0.01 --epoch=1000 --cuda_num=0