import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { ArrowLeft, ChevronLeft, ChevronRight, Loader2, Play } from "lucide-react"

function formatExpectedResult(value) {
  if (value === null || value === undefined) {
    return "No expected result recorded."
  }
  if (typeof value === "string") {
    return value
  }
  try {
    return JSON.stringify(value, null, 2)
  } catch (error) {
    return String(value)
  }
}

const clampStyle = {
  display: "-webkit-box",
  WebkitLineClamp: 3,
  WebkitBoxOrient: "vertical",
  overflow: "hidden",
}

const verticalLabelStyle = {
  writingMode: "vertical-rl",
  transform: "rotate(180deg)",
}

export default function ReferenceBenchmark() {
  const [collapsed, setCollapsed] = useState(false)
  const [selectedId, setSelectedId] = useState(null)
  const [isExecuting, setIsExecuting] = useState(!!props.executing)

  useEffect(() => {
    setIsExecuting(!!props.executing)
  }, [props.executing])

  const questions = props.questions || []
  const selected = questions.find((question) => question.id === selectedId)

  const runSelected = () => {
    if (!selected || isExecuting) return
    setIsExecuting(true)
    sendUserMessage(selected.question)
  }

  const renderBody = () => {
    if (!props.available) {
      return (
        <div className="p-4 text-sm text-muted-foreground">
          {props.unavailableMessage || "Benchmark unavailable."}
        </div>
      )
    }

    if (selected) {
      return (
        <div className="flex flex-col gap-4 p-4">
          <Button
            id="reference-benchmark-back"
            variant="ghost"
            size="sm"
            className="w-fit gap-2 px-2 text-muted-foreground hover:text-foreground"
            onClick={() => setSelectedId(null)}
          >
            <ArrowLeft className="h-4 w-4" />
            Back to questions
          </Button>

          <div className="flex flex-col gap-3 rounded-xl border bg-card p-3 shadow-sm">
            <p className="text-sm font-medium leading-snug">{selected.question}</p>
            <Button
              id="reference-benchmark-run"
              size="sm"
              className="w-full gap-2"
              disabled={isExecuting}
              onClick={runSelected}
            >
              {isExecuting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Run this question
                </>
              )}
            </Button>
          </div>

          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Golden tables
            </div>
            <div className="flex flex-col gap-2">
              {(selected.golden_tables || []).map((table, index) => (
                <div key={index} className="rounded-lg border bg-muted/40 p-3 text-xs">
                  <div className="font-semibold">{table.alias}</div>
                  <div className="mt-0.5 break-all font-mono text-[11px] text-muted-foreground">
                    {table.table_id}
                  </div>
                  {table.description ? (
                    <div className="mt-1.5 text-foreground/90">{table.description}</div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Expected result
            </div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-muted/40 p-3 text-xs">
              {formatExpectedResult(selected.expected_result)}
            </pre>
          </div>

          <div className="space-y-1.5">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Reference response
            </div>
            <p className="rounded-lg border bg-muted/40 p-3 text-sm leading-relaxed">
              {selected.reference_response || "No reference response recorded."}
            </p>
          </div>
        </div>
      )
    }

    return (
      <div className="flex flex-col gap-2 p-3">
        <div className="px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {questions.length} reference questions
        </div>
        <div className="flex max-h-[75vh] flex-col gap-1.5 overflow-y-auto pr-1">
          {questions.map((question) => (
            <button
              key={question.id}
              id={`reference-benchmark-question-${question.id}`}
              type="button"
              onClick={() => setSelectedId(question.id)}
              className="group flex items-start gap-2 rounded-lg border border-transparent p-2.5 text-left text-sm transition-colors hover:border-border hover:bg-accent hover:shadow-sm"
            >
              <span style={clampStyle} className="flex-1 leading-snug">
                {question.question}
              </span>
              <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-start">
      <button
        id="reference-benchmark-toggle"
        type="button"
        onClick={() => setCollapsed((value) => !value)}
        aria-label={collapsed ? "Expand reference benchmark" : "Collapse reference benchmark"}
        className="sticky top-3 flex w-7 shrink-0 flex-col items-center gap-3 rounded-lg border bg-muted/40 py-3 transition-colors hover:bg-accent"
      >
        {collapsed ? (
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronLeft className="h-4 w-4 text-muted-foreground" />
        )}
        <span
          style={verticalLabelStyle}
          className="whitespace-nowrap text-[10px] font-semibold uppercase tracking-wide text-muted-foreground"
        >
          Reference Benchmark
        </span>
      </button>

      {!collapsed && <div className="min-w-0 flex-1">{renderBody()}</div>}
    </div>
  )
}
