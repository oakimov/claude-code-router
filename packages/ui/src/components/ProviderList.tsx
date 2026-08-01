import { Globe, KeyRound, Pencil, ShieldCheck, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { Provider } from "@/types";
import {
  getAuthHintKey,
  getNormalizedHost,
  getProviderDescription,
  getProviderTags,
  getProviderTitle,
} from "@/lib/providerMeta";

interface ProviderListProps {
  providers: Provider[];
  onEdit: (index: number) => void;
  onRemove: (index: number) => void;
}

function EmptyProviderState() {
  const { t } = useTranslation();
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-center rounded-md border bg-card p-8 text-muted-foreground">
        {t("providers.no_providers_configured")}
      </div>
    </div>
  );
}

function ProviderIcon({ provider }: { provider: Provider }) {
  const { t } = useTranslation();
  const title = getProviderTitle(provider) || t("providers.unnamed_provider");

  if (provider.icon) {
    return (
      <img
        src={provider.icon}
        alt={`${title} icon`}
        className="h-10 w-10 rounded-md border bg-card object-contain p-1"
      />
    );
  }

  return (
    <div className="flex h-10 w-10 items-center justify-center rounded-md border bg-muted/50 text-muted-foreground">
      <Globe className="h-5 w-5" />
    </div>
  );
}

function AuthHintIcon({ hint }: { hint: string }) {
  if (hint.includes("OAuth") || hint.includes("JWT")) {
    return <ShieldCheck className="h-3 w-3" />;
  }
  return <KeyRound className="h-3 w-3" />;
}

function getLocalizedAuthHint(provider: Provider, t: (key: string) => string): string | null {
  const key = getAuthHintKey(provider);
  if (!key) return null;
  return t(`providers.auth_hint.${key}`);
}

function ModelSummary({ models }: { models: string[] }) {
  const { t } = useTranslation();

  if (models.length <= 4) {
    return (
      <div className="flex flex-wrap gap-2 pt-2">
        {models.map((model, modelIndex) => (
          <Badge
            key={modelIndex}
            variant="outline"
            className="font-normal transition-all-ease hover:scale-105"
          >
            {model || t("providers.unnamed_model")}
          </Badge>
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-wrap gap-2 pt-2">
      {models.slice(0, 3).map((model, modelIndex) => (
        <Badge
          key={modelIndex}
          variant="outline"
          className="font-normal transition-all-ease hover:scale-105"
        >
          {model || t("providers.unnamed_model")}
        </Badge>
      ))}
      <Badge variant="secondary" className="font-normal">
        {t("providers.more_models", { count: models.length - 3 })}
      </Badge>
    </div>
  );
}

function ProviderTags({ provider }: { provider: Provider }) {
  const { t } = useTranslation();
  const authHint = getLocalizedAuthHint(provider, t);
  const tags = getProviderTags(provider, authHint);
  if (tags.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2 pt-2">
      {tags.map((tag) => (
        <Badge
          key={tag}
          variant="secondary"
          className="flex items-center gap-1 font-normal"
        >
          <AuthHintIcon hint={tag} />
          {tag}
        </Badge>
      ))}
    </div>
  );
}

function ProviderCard({
  provider,
  index,
  onEdit,
  onRemove,
}: {
  provider: Provider;
  index: number;
  onEdit: (index: number) => void;
  onRemove: (index: number) => void;
}) {
  const { t } = useTranslation();
  const models = Array.isArray(provider.models) ? provider.models : [];
  const description = getProviderDescription(provider);
  const host = getNormalizedHost(provider.api_base_url) || t("providers.no_api_url");
  const title = getProviderTitle(provider) || t("providers.unnamed_provider");

  return (
    <div className="flex items-start justify-between rounded-md border bg-card p-4 transition-all hover:scale-[1.01] hover:shadow-md animate-slide-in">
      <div className="flex-1 space-y-1.5">
        <div className="flex items-start gap-3">
          <ProviderIcon provider={provider} />
          <div className="min-w-0 flex-1">
            <p className="text-md font-semibold text-foreground">{title}</p>
            <p className="text-sm text-muted-foreground">{host}</p>
            {description && <p className="text-sm text-muted-foreground">{description}</p>}
          </div>
        </div>
        <ProviderTags provider={provider} />
        <ModelSummary models={models} />
      </div>
      <div className="ml-4 flex flex-shrink-0 items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onEdit(index)}
          className="transition-all-ease hover:scale-110"
        >
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          variant="destructive"
          size="icon"
          onClick={() => onRemove(index)}
          className="transition-all duration-200 hover:scale-110"
        >
          <Trash2 className="h-4 w-4 text-current transition-colors duration-200" />
        </Button>
      </div>
    </div>
  );
}

function InvalidProviderCard({
  index,
  onEdit,
  onRemove,
}: {
  index: number;
  onEdit: (index: number) => void;
  onRemove: (index: number) => void;
}) {
  const { t } = useTranslation();

  return (
    <div className="flex items-start justify-between rounded-md border bg-card p-4 transition-all hover:scale-[1.01] hover:shadow-md animate-slide-in">
      <div className="flex-1 space-y-1.5">
        <p className="text-md font-semibold text-foreground">{t("providers.invalid_provider")}</p>
        <p className="text-sm text-muted-foreground">{t("providers.provider_data_missing")}</p>
      </div>
      <div className="ml-4 flex flex-shrink-0 items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onEdit(index)}
          className="transition-all-ease hover:scale-110"
          disabled
        >
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          variant="destructive"
          size="icon"
          onClick={() => onRemove(index)}
          className="transition-all duration-200 hover:scale-110"
        >
          <Trash2 className="h-4 w-4 text-current transition-colors duration-200" />
        </Button>
      </div>
    </div>
  );
}

export function ProviderList({ providers, onEdit, onRemove }: ProviderListProps) {
  if (!providers || !Array.isArray(providers)) {
    return <EmptyProviderState />;
  }

  return (
    <div className="space-y-3">
      {providers.map((provider, index) =>
        provider ? (
          <ProviderCard
            key={index}
            provider={provider}
            index={index}
            onEdit={onEdit}
            onRemove={onRemove}
          />
        ) : (
          <InvalidProviderCard
            key={index}
            index={index}
            onEdit={onEdit}
            onRemove={onRemove}
          />
        )
      )}
    </div>
  );
}
