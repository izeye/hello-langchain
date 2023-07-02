from html_bs_from_html_string import BSHTMLLoader

with open("example_data/fake-content.html", "r") as f:
    data = f.read()

loader = BSHTMLLoader(source="example_data/fake-content.html", html_string=data)

data = loader.load()
print(data)
