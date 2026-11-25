-- Rewrite repository-relative links (e.g. from DATA_CARD.md) to their GitHub URLs, so they
-- still work when the Markdown is rendered into the report PDF.
local REPO = "https://github.com/maxedmonds07/CS7267-LLM-Injection-Defense/blob/main/"

function Link(link)
  local target = link.target
  if target:match("^%a[%w+.-]*:") or target:match("^#") then
    return nil -- already absolute (http:, mailto:) or an in-document anchor
  end
  link.target = REPO .. target:gsub("^%./", "")
  return link
end
