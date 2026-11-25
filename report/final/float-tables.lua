-- Render every table as a floating `table` with a `tabular` inside, instead of pandoc's
-- `longtable`, so that no table breaks across pages. The tables in this paper are all short.

local function find_plain(s, needle, init)
  return s:find(needle, init or 1, true)
end

function Table(tbl)
  if not FORMAT:match("latex") then
    return nil
  end
  local tex = pandoc.write(pandoc.Pandoc({ tbl }), "latex")
  local _, open_at = find_plain(tex, "\\begin{longtable}[]{")
  if not open_at then
    return nil
  end
  -- The column spec can span lines and nest braces (p{(\linewidth - ...) * \real{0.2}}).
  local depth, spec_end = 1, open_at
  while depth > 0 do
    spec_end = spec_end + 1
    local c = tex:sub(spec_end, spec_end)
    if c == "{" then depth = depth + 1 elseif c == "}" then depth = depth - 1 end
  end
  local spec = tex:sub(open_at + 1, spec_end - 1)
  local caption = ""
  local after = spec_end + 1
  local cap_start = find_plain(tex, "\\caption{", after)
  -- A captioned table repeats its header (\endfirsthead ... \endhead); an uncaptioned one
  -- has only \endhead.
  local first_head = find_plain(tex, "\\endfirsthead", after) or find_plain(tex, "\\endhead", after)
  local header_from = after
  if cap_start and cap_start < first_head then
    local _, cap_end = find_plain(tex, "\\tabularnewline", cap_start)
    caption = tex:sub(cap_start, cap_end - #"\\tabularnewline")
    header_from = cap_end + 2
  end
  local header = tex:sub(header_from, first_head - 1)
  local _, last_foot = find_plain(tex, "\\endlastfoot\n", first_head)
  local body_end = find_plain(tex, "\\end{longtable}", last_foot)
  local body = tex:sub(last_foot + 1, body_end - 1)
  local tabular = "\\begin{tabular}{" .. spec .. "}\n" .. header .. body
    .. "\\bottomrule\\noalign{}\n\\end{tabular}\n"
  local out
  if caption ~= "" then
    out = "\\begin{table}[htbp]\n\\centering\\footnotesize\n" .. caption .. "\n" .. tabular .. "\\end{table}"
  else
    -- Uncaptioned tables (the model card's) stay in place.
    out = "\\begin{center}\\footnotesize\n" .. tabular .. "\\end{center}"
  end
  return pandoc.RawBlock("latex", out)
end
