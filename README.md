# tissue_t4
确保你已经安装好totalsegmentator相关功能，具体请参阅：
https://github.com/wasserth/TotalSegmentator；

在当前目录下新建imagesTr文件，将需要分割的文件命名为001.nii.gz、002.nii.gz、003.nii.gz等多个volume文件；
点我自动批量分割T4.bat，分割结果在labelsTr_T4文件夹下面，结果为001.nii.gz、002.nii.gz、003.nii.gz等多个mask文件；
点我自动批量分割组织.bat，分割结果在labelsTr_tissue文件夹下面，结果为001.nii.gz、002.nii.gz、003.nii.gz等多个mask文件；
###点击提取，提取结果参考如下：







labelsTr_T4文件夹下面有001.nii.gz、002.nii.gz、003.nii.gz等多个mask文件，其中每个mask文件只有一个label，label值为40，label为T4胸椎，我需要你根据T4胸椎上下高度计算每个病人（labelsTr_tissue文件夹下面有001.nii.gz、002.nii.gz、003.nii.gz等多个mask文件，每个mask文件对应有3个label，其中1为皮下脂肪、2为纵隔脂肪、3为骨骼肌）对应的每个label的体积保存最终结果为csv文件，同时需要生成相应的预览

Edit
