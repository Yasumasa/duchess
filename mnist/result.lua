----------------------------------------------------------------------
-- DS-GA 1008 Deep Learning: HW1 code submission by Duchess 
--
-- This script downloads and saves:
--
-- * best_model.t7
-- * util.lua
-- * 1_data_augmentation.lua
-- * mnist.t7.tar.gz
--
-- from http://www.cs.nyu.edu/~ym1219/FILE_NAME to the current directory.
--
-- Then, it generates "predictions.csv" which is same as our final 
-- Kaggle submission.
--
-- Usage:  
--         $th result.lua
--
--                      Feb. 2016 by Israel Malkin & Yasumasa Miyamoto
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'string'
require 'nn'

print '==> defining test procedure'

-- test function
local function testModel(test_dataset, predict_path, cpu_type)

   -- local vars
   local time = sys.clock()
   local results = nil

   -- classes
   local classes = {'1','2','3','4','5','6','7','8','9','10'}

   -- This matrix records the current confusion across classes
   confusion = optim.ConfusionMatrix(classes)  

   -- open file
   results = io.open(predict_path, "w")
   results:write("Id,Prediction\n")

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,test_dataset:size() do
      -- disp progress
      xlua.progress(t, test_dataset:size())

      -- get new sample
      local input = test_dataset.data[t]
      if cpu_type == 'double' then input = input:double() 
      elseif cpu_type == 'cuda' then input = input:cuda()  end
      local target = test_dataset.labels[t]

      -- test sample
      local pred = model:forward(input)
      pred:resize(10)
      confusion:add(pred, target)

      -- find the index of the maximum element in the output from Softmax  
      local max_elem = -math.huge
      local max_idx = nil
      for k = 1,pred:size(1) do
      	  if pred[k] > max_elem then
      	  	  max_elem = pred[k]
      	  	  max_idx = k
      	  end
      end
      
      -- write the prediction in the file
      results:write(string.format("%d,%s\n", t,max_idx))
   end

   -- close file
   results:close() 
   
   -- timing
   time = sys.clock() - time
   time = time / test_dataset:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   return confusion.totalValid * 100 
end


-- load model -----------------------------------------------------------------

print '==> load model'

model_path = 'http://www.cs.nyu.edu/~ym1219/best_model.t7'
os.execute('wget ' .. model_path)
model = torch.load('best_model.t7')
if model == nil then
	print 'model not found.'
end


-- import data loader ---------------------------------------------------------
--
-- Please use our "loadData" function in order to normalize correctly.
-- We normalized data between 0 and 1. 
--

print '==> import data loader'

-- download a utility file. This is called in 1_data_augmentation.lua.
util_path = 'http://www.cs.nyu.edu/~ym1219/util.lua'
os.execute('wget ' .. util_path)

-- download a dataloader script
dataLorder_path = 'http://www.cs.nyu.edu/~ym1219/1_data_augmentation.lua'
os.execute('wget ' .. dataLorder_path)
dofile '1_data_augmentation.lua'


-- load dataset ---------------------------------------------------------------

print '==> load dataset'

dataset_path = 'http://www.cs.nyu.edu/~ym1219/mnist.t7.tar.gz'
os.execute('wget ' .. dataset_path)
os.execute('tar -zxvf mnist.t7.tar.gz')
_,_,testData = loadData('./mnist.t7', 'full') 
print(testData)


-- run the model on the testset -----------------------------------------------
test_acc = testModel(testData, "predictions.csv", "double")

print '==> "predictions.csv" has been created!'


-- cleanup downloaded files ---------------------------------------------------

print '==> remove downloaded files'

os.execute('rm best_model.t7')
os.execute('rm util.lua')
os.execute('rm 1_data_augmentation.lua')
os.execute('rm mnist.t7.tar.gz')
os.execute('rm -r mnist.t7')

print '==> Done!'
