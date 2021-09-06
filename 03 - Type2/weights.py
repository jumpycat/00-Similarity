import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


init = torch.tensor([
    [[
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0.5,-1,0.5,0],
    [0,0,0,0,0],
    [0,0,0,0,0]],

        [
            [0, 0, 0, 0, 0],
            [0, -0.25, 0.5, -0.25, 0],
            [0, 0.5, -1, 0.5, 0],
            [0, -0.25, 0.5, -0.25, 0],
            [0, 0, 0, 0, 0]
        ],

        [
            [-0.08333, 0.16667, -0.16667, 0.16667, -0.08333],
            [0.16667, -0.5, 0.66666, -0.5, 0.16667],
            [-0.16667, 0.66666, -1, 0.66666, -0.16667],
            [0.16667, -0.5, 0.66666, -0.5, 0.16667],
            [-0.08333, 0.16667, -0.16667, 0.16667, -0.08333]
        ]],

    [[
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0.5,-1,0.5,0],
    [0,0,0,0,0],
    [0,0,0,0,0]]

    ,[
    [0,0,0,0,0],
    [0,-0.25,0.5,-0.25,0],
    [0,0.5,-1,0.5,0],
    [0,-0.25,0.5,-0.25,0],
    [0,0,0,0,0]
    ],[
    [-0.08333,0.16667,-0.16667,0.16667,-0.08333],
    [0.16667,-0.5,0.66666,-0.5,0.16667],
    [-0.16667,0.66666,-1,0.66666,-0.16667],
    [0.16667,-0.5,0.66666,-0.5,0.16667],
    [-0.08333,0.16667,-0.16667,0.16667,-0.08333]
    ]],

    [[
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0.5,-1,0.5,0],
    [0,0,0,0,0],
    [0,0,0,0,0]],[
    [0,0,0,0,0],
    [0,-0.25,0.5,-0.25,0],
    [0,0.5,-1,0.5,0],
    [0,-0.25,0.5,-0.25,0],
    [0,0,0,0,0]
    ],[
    [-0.08333,0.16667,-0.16667,0.16667,-0.08333],
    [0.16667,-0.5,0.66666,-0.5,0.16667],
    [-0.16667,0.66666,-1,0.66666,-0.16667],
    [0.16667,-0.5,0.66666,-0.5,0.16667],
    [-0.08333,0.16667,-0.16667,0.16667,-0.08333]
    ]],
])

init.cuda(device)
init = torch.transpose(init,0,1)
if __name__ == '__main__':
    print(init.shape)
    #
    print(init[0][2][0][2])
    print(init[0][1][0][2])