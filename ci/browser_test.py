# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright==1.65.2"]
# ///
"""Smoke test of the page itself, which the maths self-tests do not cover.

    uv run --with playwright==1.65.2 playwright install chromium     # once
    uv run ci/browser_test.py

The page is opened from disk and every request other than the file itself is blocked.
"""
from pathlib import Path
from playwright.sync_api import expect, sync_playwright

PAGE = (Path(__file__).resolve().parent.parent / "index.html").as_uri()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def forward_pass(page):
    expect(page.locator("#banner")).to_be_hidden()
    for stage in range(11):
        page.click(f"[data-stage='{stage}']")
        check(page.locator("#detail table.mat").count() > 0, f"stage {stage + 1} drew no matrices")
    check(page.locator("#heads .hcol").count() == 3, "expected three heads")


def training_tab(page):
    page.click("#step100")
    page.click("#tabTrain")
    for index in range(page.locator("#taskSel option").count() - 1):       # the last entry is "Your data"
        page.select_option("#taskSel", str(index))
        check(page.inner_text("#taskName").strip(), f"task {index}: no name on the Training tab")
        check(page.inner_text("#taskNote").strip(), f"task {index}: no description on the Training tab")
    page.click("#step1")
    for selector in ("#tvStep table", "#tvHistory svg", "#tvAttn svg", "#tvEmbed svg"):
        check(page.locator(selector).count() > 0, f"Training tab: nothing in {selector}")
    page.click("#tabForward")
    page.select_option("#taskSel", "0")


def guided_tour(page):
    page.click("#tourBtn")
    if page.is_visible("#tourConfirm"):
        page.click("#tourConfirmYes")
    for step in range(1, 9):
        expect(page.locator("#tourCount")).to_have_text(f"{step} of 8")
        check(page.inner_text("#tourBody").strip(), f"tour step {step} is empty")
        page.click("#tourNext")
    expect(page.locator("#tourCard")).to_be_hidden()


def weight_editing(page):
    page.click("[data-stage='1']")
    cell = page.locator("#detail td[data-w]").first
    tensor, index = cell.get_attribute("data-w"), cell.get_attribute("data-wi")
    value = lambda: page.evaluate(f"tinybert.model.list[{tensor}].t.v.data[{index}]")

    cell.scroll_into_view_if_needed()
    box = cell.bounding_box()
    before = value()
    page.mouse.move(box["x"] + 10, box["y"] + 8)
    page.mouse.down()
    page.mouse.move(box["x"] + 10, box["y"] - 40)
    page.mouse.up()
    check(value() > before, "dragging did not change the weight")
    expect(page.locator("#editOut")).to_be_visible()

    before = value()
    page.focus("#detail td[data-w][tabindex='0']")
    page.keyboard.press("Enter")
    page.keyboard.press("ArrowUp")
    page.keyboard.press("Enter")
    check(abs(value() - before - 0.02) < 1e-9, "keyboard edit did not change the weight by 0.02")
    page.keyboard.press("Enter")
    page.keyboard.press("PageUp")
    page.keyboard.press("Escape")
    check(abs(value() - before - 0.02) < 1e-9, "Esc did not undo the keyboard edit")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors, outside = [], []

        def route(request_route):
            url = request_route.request.url
            if url.startswith("file:"):
                return request_route.continue_()
            outside.append(url)
            return request_route.abort()

        page.route("**/*", route)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(PAGE)
        page.click("#tourDismiss")
        for part in (forward_pass, training_tab, guided_tour, weight_editing):
            part(page)
            check(not errors, f"{part.__name__}: page errors: {errors}")
            print(f"PASS {part.__name__}")
        check(not outside, f"the page opened from disk made outside requests: {outside}")
        print("PASS no outside requests")
        browser.close()


if __name__ == "__main__":
    main()
