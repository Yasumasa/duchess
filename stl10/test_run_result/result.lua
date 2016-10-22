----------------------------------------------------------------------
-- DS-GA 1008 Deep Learning: HW1 code submission by Duchess 
--
-- This script downloads and saves:
--
-- * HW2_best_model.net
--
-- from http://www.cs.nyu.edu/~ym1219/HW2_best_model.net to the current directory.
--
-- Please put "test.t7b" in the same folder. "test.t7b" should not be
-- normalized.
--
-- Then, it generates "predictions.csv" which is same as our final 
-- Kaggle submission.
--
-- Usage:
--         $th result.lua
--
--                      Mar. 2016 by Israel Malkin & Yasumasa Miyamoto
----------------------------------------------------------------------

----------------------------------------------------------------------
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'string'
require 'nn'
--require 'cunn'
----------------------------------------------------------------------


print '==> defining test procedure'

-- test function
local function testModel(test_dataset, predict_path)

    -- local vars
    local time = sys.clock()
    local results = nil

    -- This matrix records the current confusion across classes
    classes = {1,2,3,4,5,6,7,8,9,10}
    confusion = optim.ConfusionMatrix(classes)

    -- open file
    results = io.open(predict_path, "w")
    results:write("Id,Prediction\n")

    -- disable flips, dropouts and batch normalization
    model:evaluate()

    print('==>'.." testing")
    local bs = 4
    local t = 1
    local softmax_out = nil 
    for i=1,test_dataset.data:size(1),bs do
        xlua.progress(i, test_dataset.data:size(1)/bs)
        local outputs = model:forward(test_dataset.data:narrow(1,i,bs))
        softmax_out = nn.SoftMax():float():forward(outputs)
        confusion:batchAdd(outputs, test_dataset.labels:narrow(1,i,bs))
        prob, idx = torch.max(softmax_out, 2)

        -- write the prediction in the file
        for j=1,bs do 
            results:write(string.format("%d,%s\n", t,idx[j][1]))
            t = t + 1
        end

    end

    confusion:updateValids()
    print('test accuracy:', confusion.totalValid * 100)

    local test_acc = confusion.totalValid * 100   

    -- close file
    results:close() 

    -- timing
    time = sys.clock() - time
    time = time / test_dataset:size()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)
    confusion:zero()
end


-- load model -----------------------------------------------------------------

print '==> load model'

--model_path = 'http://www.cs.nyu.edu/~ym1219/HW2_best_model.net'
--os.execute('wget ' .. model_path)
model = torch.load('./HW2_best_model.net')
if model == nil then
    print 'model not found.'
end
print(model)

-- load dataset ---------------------------------------------------------------

print '==> load dataset'

testData = torch.load('./test.t7b')
print(testData)
testData.data = testData.data:float()
testData.labels = testData.labels:float()

--[[
dataset_path = 'http://www.cs.nyu.edu/~ym1219/test.t7b'
os.execute('wget ' .. dataset_path)
testData = torch.load('./test.t7b')
print(testData)
]]--

-- run the model on the testset -----------------------------------------------
testModel(testData, "predictions.csv")

print '==> "predictions.csv" has been created!'


-- cleanup downloaded files ---------------------------------------------------

print '==> remove downloaded files'

os.execute('rm HW2_best_model.t7')
--os.execute('rm test.t7b')

print '==> Done!'


