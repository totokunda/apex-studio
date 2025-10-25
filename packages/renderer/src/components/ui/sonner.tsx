import { useTheme } from "next-themes"
import { Toaster as Sonner } from "sonner"

type ToasterProps = React.ComponentProps<typeof Sonner>

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme()

  return (
    <Sonner
      theme={theme as ToasterProps["theme"]}
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group text-start !text-[11.5px] toast group-[.toaster]:bg-brand-background group-[.toaster]:text-brand-light group-[.toaster]:border-border group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-muted-foreground !text-[10.5px]",
          actionButton:
            "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground !text-[10.5px]",
          cancelButton:
            "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground !text-[10.5px]",
        },
      }}
      {...props}
    />
  )
}

export { Toaster }
