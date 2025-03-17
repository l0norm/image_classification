from flask import Flask, render_template
# from module import Net


app = Flask(__name__)

# model = Net()
# model.load_state_dict(torch.load(module_weights.pth,weights_only=True))
# model.eval()




@app.route('/')
def index():
    return render_template('index.html')














if __name__ == '__main__':
    app.run(debug=True)


