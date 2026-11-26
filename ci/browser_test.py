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

    # modified keys belong to the browser
    taken = page.evaluate("""() => ['ArrowLeft', 'ArrowRight', 'Enter'].filter(key => !document.activeElement.dispatchEvent(
        new KeyboardEvent('keydown', {key, altKey: true, bubbles: true, cancelable: true})))""")
    check(not taken, f"Alt shortcuts were swallowed on a grid cell: {taken}")

    # starting an edit during training pauses it and keeps the tooltip beside the cell
    page.click("#trainBtn")
    page.focus("#detail td[data-w][tabindex='0']")
    page.keyboard.press("Enter")
    expect(page.locator("#tip")).to_be_visible()
    cell_box = page.locator("#detail td.editing").first.bounding_box()
    tip_box = page.locator("#tip").bounding_box()
    check(not page.evaluate("tinybert.state.trainer.running"), "editing did not pause training")
    check(abs(tip_box["x"] - cell_box["x"]) < 400 and tip_box["x"] > 20, f"tooltip is away from the cell: {tip_box}")

    # a redraw that removes the edited cell ends the edit, even without a focus event
    page.evaluate("""() => { addEventListener('focusout', e => e.stopPropagation(), {capture: true, once: false});
        tinybert.state.stage = 0; document.querySelector("[data-stage='4']").click(); }""")
    check(not page.evaluate("tinybert.state.drag"), "the edit stayed open after its cell was removed")

    # Training tab: Enter and Space select the focused item
    page.click("#tabTrain")
    page.click("#step1")
    for key, row in (("Enter", 2), (" ", 3)):
        page.focus(f"#tvStep tr[data-tensor='{row}']")
        page.keyboard.press(key)
        check(page.evaluate("tinybert.state.tv.tensor") == row, f"{key!r} did not select tensor row {row}")


def story(page):
    page.click("#tabForward")
    page.click("[data-stage='2']")
    page.click("#storyBtn")
    expect(page.locator("#story")).to_be_visible()
    check(page.locator("#story .msg").count() > 6, "the story popup has no conversation")
    page.keyboard.press("ArrowRight")
    check(page.evaluate("tinybert.state.stage") == 2, "arrow keys stepped the pipeline behind the popup")
    page.keyboard.press("Escape")
    expect(page.locator("#story")).to_be_hidden()
    page.click("#storyBtn")
    page.click("#storyClose")
    expect(page.locator("#story")).to_be_hidden()

    # dragging a text selection off the popup keeps it open; a click outside closes it
    page.click("#storyBtn")
    box = page.locator("#story .msg").first.bounding_box()
    page.mouse.move(box["x"] + 20, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(3, 400)
    page.mouse.up()
    expect(page.locator("#story")).to_be_visible()
    page.mouse.click(3, 400)
    expect(page.locator("#story")).to_be_hidden()

    # during the tour, arrows and Esc belong to the popup (the modal blocks a real click on the link)
    page.click("#tourBtn")
    if page.is_visible("#tourConfirm"):
        page.click("#tourConfirmYes")
    page.click("#storyBtn")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("Escape")
    expect(page.locator("#story")).to_be_hidden()
    expect(page.locator("#tourCount")).to_have_text("1 of 8")
    page.click("#tourSkip")


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
        for part in (forward_pass, training_tab, guided_tour, weight_editing, story):
            part(page)
            check(not errors, f"{part.__name__}: page errors: {errors}")
            print(f"PASS {part.__name__}")
        check(not outside, f"the page opened from disk made outside requests: {outside}")
        print("PASS no outside requests")
        browser.close()


if __name__ == "__main__":
    main()
