import os
# 功能：重新排序并生成文件名如“00000001”
def rename(dir):
    # 补零到z位
    z=5
    # 判断是否为文件夹
    if os.path.isdir(dir):
        fileList=os.listdir(dir)
        fileNum=0
        for file in fileList:
            # 判断是否为文件
            if os.path.isfile(dir+"\\"+file):
                fileNum+=1
                for i in range(len(file)-1,-1,-1):
                    if file[i]==".":
                        posDot=i
                        break
                curFile=str(fileNum).zfill(z)+file[posDot:]
                os.rename(dir+"\\"+file,dir+"\\"+curFile)
                print(file+"已重命名为"+curFile)
        print("该路径下共有"+str(fileNum)+"个文件与"+str(len(fileList)-fileNum)+"个文件夹，现已对文件进行重命名操作")
    else:
        print("请输入正确的文件路径")
rename("../HSVselector/crossroads20230802")