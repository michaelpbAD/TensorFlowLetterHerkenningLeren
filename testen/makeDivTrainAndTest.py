import os
import shutil

#indir = '..\\..\\dataset\\English\\Fnt'
indir = '..\\..\\dataset\\English\\\Hnd\\Img'

traindir = '..\\..\\dataset\\Train'
testdir = '..\\..\\dataset\\Test'

if not os.path.exists(traindir):
    os.mkdir(traindir)
if not os.path.exists(testdir):
    os.mkdir(testdir)

# for root, dirs, files in os.walk(indir):
#
#     for name in dirs:
#
#         mNR = int(name[-2:])
#         #hooftletter bij klijne letter toevoegen
#         if mNR > 36:
#             mNR -= 26
#
#         trainDst_dir = os.path.join(traindir, str(mNR))
#         if not os.path.exists(trainDst_dir):
#             os.mkdir(trainDst_dir)
#         print(trainDst_dir)
#
#         testDst_dir = os.path.join(testdir, str(mNR))
#         if not os.path.exists(testDst_dir):
#             os.mkdir(testDst_dir)
#         print(testDst_dir)
#
#         trainTel = len(os.listdir(trainDst_dir))
#         testTel = len(os.listdir(testDst_dir))
#         tel = 0
#         for root2, dirs2, files2 in os.walk(os.path.join(root, name)):
#             for name2 in files2:
#                 tel += 1
#                 # x% test fille
#                 if (tel % 10):
#                     trainTel += 1
#                     dst_dir = trainDst_dir
#
#                     # file copy
#                     src_file = os.path.join(root2, name2)
#                     shutil.copy(src_file, dst_dir)
#
#                     # file rename
#                     filename, file_extension = os.path.splitext(name2)
#                     dst_file = os.path.join(dst_dir, name2)
#                     new_dst_file_name = os.path.join(dst_dir, str(trainTel).zfill(4) + file_extension)
#                     os.rename(dst_file, new_dst_file_name)
#
#                     print(new_dst_file_name)
#                 else:
#                     testTel += 1
#                     dst_dir = testDst_dir
#
#                     # file copy
#                     src_file = os.path.join(root2, name2)
#                     shutil.copy(src_file, dst_dir)
#
#                     # file rename
#                     filename, file_extension = os.path.splitext(name2)
#                     dst_file = os.path.join(dst_dir, name2)
#                     new_dst_file_name = os.path.join(dst_dir, str(testTel).zfill(4) + file_extension)
#                     os.rename(dst_file, new_dst_file_name)
#
#                     print(new_dst_file_name)


print('train tot:'+str(len([name for name in os.listdir(traindir) for name in os.listdir(os.path.join(traindir, name))])))

print('test tot:'+str(len([name for name in os.listdir(testdir) for name in os.listdir(os.path.join(testdir, name))])))
