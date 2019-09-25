import base64
from io import BytesIO


def plot_to_html_image(plt):
    img = BytesIO()
    plt.savefig(img, transparent=True, bbox_inches='tight')
    img.seek(0)
    plt.close()
    html_string = '<img src="data:image/png;base64,' + base64.b64encode(img.read()).decode("UTF-8") + '"/>'
    return html_string