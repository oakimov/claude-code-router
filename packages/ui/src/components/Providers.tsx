import { useMemo, useRef, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { useConfig } from "./ConfigProvider";
import { ProviderList } from "./ProviderList";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Info, X, Trash2, Plus, Eye, EyeOff, Search, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Combobox } from "@/components/ui/combobox";
import { ComboInput } from "@/components/ui/combo-input";
import { api } from "@/lib/api";
import type { Provider } from "@/types";
import {
  getAuthHintKey,
  getNormalizedHost,
  getProviderDescription,
  getProviderTags,
  getProviderTitle,
  getTemplateOptionLabel,
} from "@/lib/providerMeta";

interface ProviderType extends Provider {}

function ProviderTemplatePreview({ provider }: { provider: Provider | null }) {
  const { t } = useTranslation();
  if (!provider) return null;

  const title = getProviderTitle(provider) || t("providers.unnamed_provider");
  const description = getProviderDescription(provider);
  const host = getNormalizedHost(provider.api_base_url) || t("providers.no_api_url");
  const authHintKey = getAuthHintKey(provider);
  const authHint = authHintKey ? t(`providers.auth_hint.${authHintKey}`) : null;
  const tags = getProviderTags(provider, authHint);

  return (
    <div className="space-y-2 rounded-md border bg-gray-50 p-3">
      <div className="flex items-start gap-3">
        {provider.icon ? (
          <img
            src={provider.icon}
            alt={`${title} icon`}
            className="h-10 w-10 rounded-md border bg-white object-contain p-1"
          />
        ) : (
          <div className="flex h-10 w-10 items-center justify-center rounded-md border bg-white text-gray-400">
            <Info className="h-4 w-4" />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="font-medium text-gray-900">{title}</div>
          <div className="text-sm text-gray-500">{host}</div>
          {description && <div className="mt-1 text-sm text-gray-600">{description}</div>}
        </div>
      </div>
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {tags.map((tag) => (
            <Badge key={tag} variant="secondary" className="font-normal">
              {tag}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

export function Providers() {
  const { t } = useTranslation();
  const { config, setConfig } = useConfig();
  const [editingProviderIndex, setEditingProviderIndex] = useState<number | null>(null);
  const [deletingProviderIndex, setDeletingProviderIndex] = useState<number | null>(null);
  const [hasFetchedModels, setHasFetchedModels] = useState<Record<number, boolean>>({});
  const [providerParamInputs, setProviderParamInputs] = useState<Record<string, { name: string; value: string }>>({});
  const [modelParamInputs, setModelParamInputs] = useState<Record<string, { name: string; value: string }>>({});
  const [availableTransformers, setAvailableTransformers] = useState<{ name: string; endpoint: string | null }[]>([]);
  const [editingProviderData, setEditingProviderData] = useState<ProviderType | null>(null);
  const [isNewProvider, setIsNewProvider] = useState<boolean>(false);
  const [providerTemplates, setProviderTemplates] = useState<ProviderType[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<ProviderType | null>(null);
  const [showApiKey, setShowApiKey] = useState<Record<number, boolean>>({});
  const [apiKeyError, setApiKeyError] = useState<string | null>(null);
  const [nameError, setNameError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const comboInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchProviderTemplates = async () => {
      try {
        const response = await fetch("https://pub-0dc3e1677e894f07bbea11b17a29e032.r2.dev/providers.json");
        if (response.ok) {
          const data = await response.json();
          setProviderTemplates(Array.isArray(data) ? data : []);
        } else {
          console.error("Failed to fetch provider templates");
        }
      } catch (error) {
        console.error("Failed to fetch provider templates:", error);
      }
    };

    fetchProviderTemplates();
  }, []);

  useEffect(() => {
    const fetchTransformers = async () => {
      try {
        const response = await api.get<{ transformers: { name: string; endpoint: string | null }[] }>("/transformers");
        setAvailableTransformers(response.transformers);
      } catch (error) {
        console.error("Failed to fetch transformers:", error);
      }
    };

    fetchTransformers();
  }, []);

  if (!config) {
    return (
      <Card className="flex h-full flex-col rounded-lg border shadow-sm">
        <CardHeader className="flex flex-row items-center justify-between border-b p-4">
          <CardTitle className="text-lg">{t("providers.title")}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-grow items-center justify-center p-4">
          <div className="text-gray-500">Loading providers configuration...</div>
        </CardContent>
      </Card>
    );
  }

  const validProviders = Array.isArray(config.Providers) ? config.Providers : [];

  const templateOptions = useMemo(
    () =>
      providerTemplates.map((provider) => ({
        label: getTemplateOptionLabel(provider, t("providers.no_api_url")),
        value: JSON.stringify(provider),
      })),
    [providerTemplates, t]
  );

  const editingProviderSummary = useMemo(() => {
    if (!editingProviderData) return null;
    const authHintKey = getAuthHintKey(editingProviderData);
    const authHint = authHintKey ? t(`providers.auth_hint.${authHintKey}`) : null;
    return {
      host: getNormalizedHost(editingProviderData.api_base_url) || t("providers.no_api_url"),
      description: getProviderDescription(editingProviderData),
      tags: getProviderTags(editingProviderData, authHint),
    };
  }, [editingProviderData, t]);

  const handleAddProvider = () => {
    const newProvider: ProviderType = { name: "", api_base_url: "", api_key: "", models: [] };
    setEditingProviderIndex(config.Providers.length);
    setEditingProviderData(newProvider);
    setSelectedTemplate(null);
    setIsNewProvider(true);
    setShowApiKey((prev) => ({
      ...prev,
      [config.Providers.length]: false,
    }));
    setApiKeyError(null);
    setNameError(null);
  };

  const handleEditProvider = (index: number) => {
    const actualIndex = validProviders.indexOf(filteredProviders[index]);
    const provider = config.Providers[actualIndex];
    setEditingProviderIndex(actualIndex);
    setEditingProviderData(JSON.parse(JSON.stringify(provider)));
    setSelectedTemplate(null);
    setIsNewProvider(false);
    setShowApiKey((prev) => ({
      ...prev,
      [actualIndex]: false,
    }));
    setApiKeyError(null);
    setNameError(null);
  };

  const handleSaveProvider = () => {
    if (!editingProviderData) return;

    if (!editingProviderData.name || editingProviderData.name.trim() === "") {
      setNameError(t("providers.name_required"));
      return;
    }

    const trimmedName = editingProviderData.name.trim();
    const isDuplicate = config.Providers.some((provider, index) => {
      if (!isNewProvider && index === editingProviderIndex) {
        return false;
      }
      return provider.name.toLowerCase() === trimmedName.toLowerCase();
    });

    if (isDuplicate) {
      setNameError(t("providers.name_duplicate"));
      return;
    }

    if (!editingProviderData.api_key || editingProviderData.api_key.trim() === "") {
      setApiKeyError(t("providers.api_key_required"));
      return;
    }

    setApiKeyError(null);
    setNameError(null);

    if (editingProviderIndex !== null) {
      const newProviders = [...config.Providers];
      if (isNewProvider) {
        newProviders.push(editingProviderData);
      } else {
        newProviders[editingProviderIndex] = editingProviderData;
      }
      setConfig({ ...config, Providers: newProviders });
    }

    if (editingProviderIndex !== null) {
      setShowApiKey((prev) => {
        const newState = { ...prev };
        delete newState[editingProviderIndex];
        return newState;
      });
    }
    setEditingProviderIndex(null);
    setEditingProviderData(null);
    setSelectedTemplate(null);
    setIsNewProvider(false);
  };

  const handleCancelAddProvider = () => {
    if (editingProviderIndex !== null) {
      setHasFetchedModels((prev) => {
        const newState = { ...prev };
        delete newState[editingProviderIndex];
        return newState;
      });
      setShowApiKey((prev) => {
        const newState = { ...prev };
        delete newState[editingProviderIndex];
        return newState;
      });
    }
    setEditingProviderIndex(null);
    setEditingProviderData(null);
    setSelectedTemplate(null);
    setIsNewProvider(false);
    setApiKeyError(null);
    setNameError(null);
  };

  const handleSetDeletingProviderIndex = (filteredIndex: number) => {
    setDeletingProviderIndex(filteredIndex);
  };

  const handleRemoveProvider = (filteredIndex: number) => {
    const actualIndex = validProviders.indexOf(filteredProviders[filteredIndex]);
    const newProviders = [...config.Providers];
    newProviders.splice(actualIndex, 1);
    setConfig({ ...config, Providers: newProviders });
    setDeletingProviderIndex(null);
  };

  const handleProviderChange = (_index: number, field: string, value: string) => {
    if (editingProviderData) {
      const updatedProvider = { ...editingProviderData, [field]: value };
      setEditingProviderData(updatedProvider);
    }
  };

  const handleProviderTransformerChange = (_index: number, transformerPath: string) => {
    if (!transformerPath || !editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (!updatedProvider.transformer) {
      updatedProvider.transformer = { use: [] };
    }

    updatedProvider.transformer.use = [...updatedProvider.transformer.use, transformerPath];
    setEditingProviderData(updatedProvider);
  };

  const removeProviderTransformerAtIndex = (_index: number, transformerIndex: number) => {
    if (!editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (updatedProvider.transformer) {
      const newUseArray = [...updatedProvider.transformer.use];
      newUseArray.splice(transformerIndex, 1);
      updatedProvider.transformer.use = newUseArray;

      if (newUseArray.length === 0 && Object.keys(updatedProvider.transformer).length === 1) {
        delete updatedProvider.transformer;
      }
    }

    setEditingProviderData(updatedProvider);
  };

  const handleModelTransformerChange = (_providerIndex: number, model: string, transformerPath: string) => {
    if (!transformerPath || !editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (!updatedProvider.transformer) {
      updatedProvider.transformer = { use: [] };
    }

    if (!updatedProvider.transformer[model]) {
      updatedProvider.transformer[model] = { use: [] };
    }

    updatedProvider.transformer[model].use = [...updatedProvider.transformer[model].use, transformerPath];
    setEditingProviderData(updatedProvider);
  };

  const removeModelTransformerAtIndex = (_providerIndex: number, model: string, transformerIndex: number) => {
    if (!editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (updatedProvider.transformer && updatedProvider.transformer[model]) {
      const newUseArray = [...updatedProvider.transformer[model].use];
      newUseArray.splice(transformerIndex, 1);
      updatedProvider.transformer[model].use = newUseArray;

      if (newUseArray.length === 0 && Object.keys(updatedProvider.transformer[model]).length === 1) {
        delete updatedProvider.transformer[model];
      }
    }

    setEditingProviderData(updatedProvider);
  };

  const addProviderTransformerParameter = (_providerIndex: number, transformerIndex: number, paramName: string, paramValue: string) => {
    if (!editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (!updatedProvider.transformer) {
      updatedProvider.transformer = { use: [] };
    }

    if (updatedProvider.transformer.use && updatedProvider.transformer.use.length > transformerIndex) {
      const targetTransformer = updatedProvider.transformer.use[transformerIndex];

      if (Array.isArray(targetTransformer)) {
        const transformerArray = [...targetTransformer];
        if (transformerArray.length > 1 && typeof transformerArray[1] === "object" && transformerArray[1] !== null) {
          const existingParams = transformerArray[1] as Record<string, unknown>;
          const paramsObj: Record<string, unknown> = { ...existingParams, [paramName]: paramValue };
          transformerArray[1] = paramsObj;
        } else if (transformerArray.length > 1) {
          const paramsObj = { [paramName]: paramValue };
          transformerArray.splice(1, transformerArray.length - 1, paramsObj);
        } else {
          const paramsObj = { [paramName]: paramValue };
          transformerArray.push(paramsObj);
        }

        updatedProvider.transformer.use[transformerIndex] = transformerArray as string | (string | Record<string, unknown> | { max_tokens: number })[];
      } else {
        const paramsObj = { [paramName]: paramValue };
        updatedProvider.transformer.use[transformerIndex] = [targetTransformer as string, paramsObj];
      }
    }

    setEditingProviderData(updatedProvider);
  };

  const removeProviderTransformerParameterAtIndex = (_providerIndex: number, transformerIndex: number, paramName: string) => {
    if (!editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (!updatedProvider.transformer?.use || updatedProvider.transformer.use.length <= transformerIndex) {
      return;
    }

    const targetTransformer = updatedProvider.transformer.use[transformerIndex];
    if (Array.isArray(targetTransformer) && targetTransformer.length > 1) {
      const transformerArray = [...targetTransformer];
      if (typeof transformerArray[1] === "object" && transformerArray[1] !== null) {
        const paramsObj = { ...(transformerArray[1] as Record<string, unknown>) };
        delete paramsObj[paramName];

        if (Object.keys(paramsObj).length === 0) {
          transformerArray.splice(1, 1);
        } else {
          transformerArray[1] = paramsObj;
        }

        updatedProvider.transformer.use[transformerIndex] = transformerArray;
        setEditingProviderData(updatedProvider);
      }
    }
  };

  const addModelTransformerParameter = (_providerIndex: number, model: string, transformerIndex: number, paramName: string, paramValue: string) => {
    if (!editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (!updatedProvider.transformer) {
      updatedProvider.transformer = { use: [] };
    }

    if (!updatedProvider.transformer[model]) {
      updatedProvider.transformer[model] = { use: [] };
    }

    if (updatedProvider.transformer[model].use && updatedProvider.transformer[model].use.length > transformerIndex) {
      const targetTransformer = updatedProvider.transformer[model].use[transformerIndex];

      if (Array.isArray(targetTransformer)) {
        const transformerArray = [...targetTransformer];
        if (transformerArray.length > 1 && typeof transformerArray[1] === "object" && transformerArray[1] !== null) {
          const existingParams = transformerArray[1] as Record<string, unknown>;
          const paramsObj: Record<string, unknown> = { ...existingParams, [paramName]: paramValue };
          transformerArray[1] = paramsObj;
        } else if (transformerArray.length > 1) {
          const paramsObj = { [paramName]: paramValue };
          transformerArray.splice(1, transformerArray.length - 1, paramsObj);
        } else {
          const paramsObj = { [paramName]: paramValue };
          transformerArray.push(paramsObj);
        }

        updatedProvider.transformer[model].use[transformerIndex] = transformerArray as string | (string | Record<string, unknown> | { max_tokens: number })[];
      } else {
        const paramsObj = { [paramName]: paramValue };
        updatedProvider.transformer[model].use[transformerIndex] = [targetTransformer as string, paramsObj];
      }
    }

    setEditingProviderData(updatedProvider);
  };

  const removeModelTransformerParameterAtIndex = (_providerIndex: number, model: string, transformerIndex: number, paramName: string) => {
    if (!editingProviderData) return;

    const updatedProvider = { ...editingProviderData };

    if (!updatedProvider.transformer?.[model]?.use || updatedProvider.transformer[model].use.length <= transformerIndex) {
      return;
    }

    const targetTransformer = updatedProvider.transformer[model].use[transformerIndex];
    if (Array.isArray(targetTransformer) && targetTransformer.length > 1) {
      const transformerArray = [...targetTransformer];
      if (typeof transformerArray[1] === "object" && transformerArray[1] !== null) {
        const paramsObj = { ...(transformerArray[1] as Record<string, unknown>) };
        delete paramsObj[paramName];

        if (Object.keys(paramsObj).length === 0) {
          transformerArray.splice(1, 1);
        } else {
          transformerArray[1] = paramsObj;
        }

        updatedProvider.transformer[model].use[transformerIndex] = transformerArray;
        setEditingProviderData(updatedProvider);
      }
    }
  };

  const handleAddModel = (_index: number, model: string) => {
    if (!model.trim() || !editingProviderData) return;

    const updatedProvider = { ...editingProviderData };
    const models = Array.isArray(updatedProvider.models) ? [...updatedProvider.models] : [];

    if (!models.includes(model.trim())) {
      models.push(model.trim());
      updatedProvider.models = models;
      setEditingProviderData(updatedProvider);
    }
  };

  const handleTemplateImport = (value: string) => {
    if (!value) return;
    try {
      const parsedTemplate = JSON.parse(value) as ProviderType;
      if (parsedTemplate) {
        setSelectedTemplate(parsedTemplate);
        const currentName = editingProviderData?.name;
        const newProviderData = JSON.parse(JSON.stringify(parsedTemplate));

        if (!isNewProvider && currentName) {
          newProviderData.name = currentName;
        }

        setEditingProviderData(newProviderData as ProviderType);
      }
    } catch (e) {
      console.error("Failed to parse template", e);
    }
  };

  const handleRemoveModel = (_providerIndex: number, modelIndex: number) => {
    if (!editingProviderData) return;

    const updatedProvider = { ...editingProviderData };
    const models = Array.isArray(updatedProvider.models) ? [...updatedProvider.models] : [];

    if (modelIndex >= 0 && modelIndex < models.length) {
      models.splice(modelIndex, 1);
      updatedProvider.models = models;
      setEditingProviderData(updatedProvider);
    }
  };

  const editingProvider =
    editingProviderData ||
    (editingProviderIndex !== null ? validProviders[editingProviderIndex] : null);

  const filteredProviders = validProviders.filter((provider) => {
    if (!searchTerm) return true;
    const term = searchTerm.toLowerCase();
    if (
      (provider.name && provider.name.toLowerCase().includes(term)) ||
      (provider.api_base_url && provider.api_base_url.toLowerCase().includes(term))
    ) {
      return true;
    }
    if (provider.models && Array.isArray(provider.models)) {
      return provider.models.some((model) => model && model.toLowerCase().includes(term));
    }
    return false;
  });

  return (
    <Card className="flex h-full flex-col rounded-lg border shadow-sm">
      <CardHeader className="flex flex-col gap-3 border-b p-4">
        <div className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg">
            {t("providers.title")} <span className="text-sm font-normal text-gray-500">({filteredProviders.length}/{validProviders.length})</span>
          </CardTitle>
          <Button onClick={handleAddProvider}>{t("providers.add")}</Button>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
            <Input
              placeholder={t("providers.search")}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8"
            />
          </div>
          {searchTerm && (
            <Button variant="ghost" size="icon" onClick={() => setSearchTerm("") }>
              <XCircle className="h-4 w-4" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-grow overflow-y-auto p-4">
        <ProviderList
          providers={filteredProviders}
          onEdit={handleEditProvider}
          onRemove={handleSetDeletingProviderIndex}
        />
      </CardContent>

      <Dialog
        open={editingProviderIndex !== null}
        onOpenChange={(open) => {
          if (!open) {
            handleCancelAddProvider();
          }
        }}
      >
        <DialogContent className="max-h-[80vh] flex flex-col sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("providers.edit")}</DialogTitle>
          </DialogHeader>
          {editingProvider && editingProviderIndex !== null && (
            <div className="space-y-4 flex-grow overflow-y-auto p-4">
              {providerTemplates.length > 0 && (
                <div className="space-y-2">
                  <Label>{t("providers.import_from_template")}</Label>
                  <Combobox
                    options={templateOptions}
                    value={selectedTemplate ? JSON.stringify(selectedTemplate) : ""}
                    onChange={handleTemplateImport}
                    placeholder={t("providers.select_template")}
                    emptyPlaceholder={t("providers.no_templates_found")}
                  />
                  <ProviderTemplatePreview provider={selectedTemplate} />
                </div>
              )}

              {editingProviderSummary && (
                <div className="space-y-2 rounded-md border bg-blue-50/40 p-3">
                  <div className="text-sm font-medium text-gray-900">{t("providers.current_provider_summary")}</div>
                  <div className="text-sm text-gray-600">{editingProviderSummary.host}</div>
                  {editingProviderSummary.description && (
                    <div className="text-sm text-gray-600">{editingProviderSummary.description}</div>
                  )}
                  {editingProviderSummary.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {editingProviderSummary.tags.map((tag) => (
                        <Badge key={tag} variant="secondary" className="font-normal">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="name">{t("providers.name")}</Label>
                <Input
                  id="name"
                  value={editingProvider.name || ""}
                  onChange={(e) => {
                    handleProviderChange(editingProviderIndex, "name", e.target.value);
                    if (nameError) {
                      setNameError(null);
                    }
                  }}
                  className={nameError ? "border-red-500" : ""}
                />
                {nameError && <p className="text-sm text-red-500">{nameError}</p>}
              </div>
              <div className="space-y-2">
                <Label htmlFor="api_base_url">{t("providers.api_base_url")}</Label>
                <Input
                  id="api_base_url"
                  value={editingProvider.api_base_url || ""}
                  onChange={(e) => handleProviderChange(editingProviderIndex, "api_base_url", e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="api_key">{t("providers.api_key")}</Label>
                <div className="relative">
                  <Input
                    id="api_key"
                    type={showApiKey[editingProviderIndex || 0] ? "text" : "password"}
                    value={editingProvider.api_key || ""}
                    onChange={(e) => handleProviderChange(editingProviderIndex, "api_key", e.target.value)}
                    className={apiKeyError ? "border-red-500" : ""}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="absolute right-2 top-1/2 h-8 w-8 -translate-y-1/2 transform"
                    onClick={() => {
                      const index = editingProviderIndex || 0;
                      setShowApiKey((prev) => ({
                        ...prev,
                        [index]: !prev[index],
                      }));
                    }}
                  >
                    {showApiKey[editingProviderIndex || 0] ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                {apiKeyError && <p className="text-sm text-red-500">{apiKeyError}</p>}
              </div>
              <div className="space-y-2">
                <Label htmlFor="models">{t("providers.models")}</Label>
                <div className="space-y-2">
                  <div className="flex gap-2">
                    <div className="flex-1">
                      {hasFetchedModels[editingProviderIndex] ? (
                        <ComboInput
                          ref={comboInputRef}
                          options={(editingProvider.models || []).map((model: string) => ({ label: model, value: model }))}
                          value=""
                          onChange={() => {
                            // Only update input values, do not add models
                          }}
                          onEnter={(value) => {
                            if (editingProviderIndex !== null) {
                              handleAddModel(editingProviderIndex, value);
                            }
                          }}
                          inputPlaceholder={t("providers.models_placeholder")}
                        />
                      ) : (
                        <Input
                          id="models"
                          placeholder={t("providers.models_placeholder")}
                          onKeyDown={(e) => {
                            if (
                              e.key === "Enter" &&
                              e.currentTarget.value.trim() &&
                              editingProviderIndex !== null
                            ) {
                              handleAddModel(editingProviderIndex, e.currentTarget.value);
                              e.currentTarget.value = "";
                            }
                          }}
                        />
                      )}
                    </div>
                    <Button
                      onClick={() => {
                        if (hasFetchedModels[editingProviderIndex] && comboInputRef.current) {
                          const comboInput = comboInputRef.current as unknown as {
                            getCurrentValue(): string;
                            clearInput(): void;
                          };
                          const currentValue = comboInput.getCurrentValue();
                          if (
                            currentValue &&
                            currentValue.trim() &&
                            editingProviderIndex !== null
                          ) {
                            handleAddModel(editingProviderIndex, currentValue.trim());
                            comboInput.clearInput();
                          }
                        } else {
                          const input = document.getElementById("models") as HTMLInputElement;
                          if (
                            input &&
                            input.value.trim() &&
                            editingProviderIndex !== null
                          ) {
                            handleAddModel(editingProviderIndex, input.value);
                            input.value = "";
                          }
                        }
                      }}
                    >
                      {t("providers.add_model")}
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 pt-2">
                    {(editingProvider.models || []).map((model: string, modelIndex: number) => (
                      <Badge key={modelIndex} variant="outline" className="flex items-center gap-1 font-normal">
                        {model}
                        <button
                          type="button"
                          className="ml-1 rounded-full hover:bg-gray-200"
                          onClick={() =>
                            editingProviderIndex !== null &&
                            handleRemoveModel(editingProviderIndex, modelIndex)
                          }
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <Label>{t("providers.provider_transformer")}</Label>
                <div className="flex gap-2">
                  <Combobox
                    options={availableTransformers.map((transformer) => ({
                      label: transformer.name,
                      value: transformer.name,
                    }))}
                    value=""
                    onChange={(value) => {
                      if (editingProviderIndex !== null) {
                        handleProviderTransformerChange(editingProviderIndex, value);
                      }
                    }}
                    placeholder={t("providers.select_transformer")}
                    emptyPlaceholder={t("providers.no_transformers")}
                  />
                </div>

                {editingProvider.transformer?.use && editingProvider.transformer.use.length > 0 && (
                  <div className="mt-2 space-y-2">
                    <div className="text-sm font-medium text-gray-700">
                      {t("providers.selected_transformers")}
                    </div>
                    {editingProvider.transformer.use.map(
                      (
                        transformer: string | (string | Record<string, unknown> | { max_tokens: number })[],
                        transformerIndex: number
                      ) => (
                        <div key={transformerIndex} className="rounded-md border p-3">
                          <div className="mb-2 flex items-center gap-2">
                            <div className="flex-1 rounded bg-gray-50 p-2 text-sm">
                              {typeof transformer === "string"
                                ? transformer
                                : Array.isArray(transformer)
                                  ? String(transformer[0])
                                  : String(transformer)}
                            </div>
                            <Button
                              variant="outline"
                              size="icon"
                              onClick={() => {
                                if (editingProviderIndex !== null) {
                                  removeProviderTransformerAtIndex(
                                    editingProviderIndex,
                                    transformerIndex
                                  );
                                }
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>

                          <div className="mt-2 border-l-2 border-gray-200 pl-4">
                            <Label className="text-sm">
                              {t("providers.transformer_parameters")}
                            </Label>
                            <div className="mt-1 space-y-2">
                              <div className="flex gap-2">
                                <Input
                                  placeholder={t("providers.parameter_name")}
                                  value={
                                    providerParamInputs[
                                      `provider-${editingProviderIndex}-transformer-${transformerIndex}`
                                    ]?.name || ""
                                  }
                                  onChange={(e) => {
                                    const key = `provider-${editingProviderIndex}-transformer-${transformerIndex}`;
                                    setProviderParamInputs((prev) => ({
                                      ...prev,
                                      [key]: {
                                        ...(prev[key] || { name: "", value: "" }),
                                        name: e.target.value,
                                      },
                                    }));
                                  }}
                                />
                                <Input
                                  placeholder={t("providers.parameter_value")}
                                  value={
                                    providerParamInputs[
                                      `provider-${editingProviderIndex}-transformer-${transformerIndex}`
                                    ]?.value || ""
                                  }
                                  onChange={(e) => {
                                    const key = `provider-${editingProviderIndex}-transformer-${transformerIndex}`;
                                    setProviderParamInputs((prev) => ({
                                      ...prev,
                                      [key]: {
                                        ...(prev[key] || { name: "", value: "" }),
                                        value: e.target.value,
                                      },
                                    }));
                                  }}
                                />
                                <Button
                                  size="sm"
                                  onClick={() => {
                                    if (editingProviderIndex !== null) {
                                      const key = `provider-${editingProviderIndex}-transformer-${transformerIndex}`;
                                      const paramInput = providerParamInputs[key];
                                      if (
                                        paramInput &&
                                        paramInput.name &&
                                        paramInput.value
                                      ) {
                                        addProviderTransformerParameter(
                                          editingProviderIndex,
                                          transformerIndex,
                                          paramInput.name,
                                          paramInput.value
                                        );
                                        setProviderParamInputs((prev) => ({
                                          ...prev,
                                          [key]: { name: "", value: "" },
                                        }));
                                      }
                                    }
                                  }}
                                >
                                  <Plus className="h-4 w-4" />
                                </Button>
                              </div>

                              {(() => {
                                if (
                                  !editingProvider.transformer?.use ||
                                  editingProvider.transformer.use.length <= transformerIndex
                                ) {
                                  return null;
                                }

                                const targetTransformer =
                                  editingProvider.transformer.use[transformerIndex];
                                let params = {};

                                if (
                                  Array.isArray(targetTransformer) &&
                                  targetTransformer.length > 1 &&
                                  typeof targetTransformer[1] === "object" &&
                                  targetTransformer[1] !== null
                                ) {
                                  params = targetTransformer[1] as Record<string, unknown>;
                                }

                                return Object.keys(params).length > 0 ? (
                                  <div className="space-y-1">
                                    {Object.entries(params).map(([key, value]) => (
                                      <div
                                        key={key}
                                        className="flex items-center justify-between rounded bg-gray-50 p-2"
                                      >
                                        <div className="text-sm">
                                          <span className="font-medium">{key}:</span> {String(value)}
                                        </div>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          className="h-6 w-6 p-0"
                                          onClick={() => {
                                            if (editingProviderIndex !== null) {
                                              removeProviderTransformerParameterAtIndex(
                                                editingProviderIndex,
                                                transformerIndex,
                                                key
                                              );
                                            }
                                          }}
                                        >
                                          <X className="h-3 w-3" />
                                        </Button>
                                      </div>
                                    ))}
                                  </div>
                                ) : null;
                              })()}
                            </div>
                          </div>
                        </div>
                      )
                    )}
                  </div>
                )}
              </div>

              {editingProvider.models && editingProvider.models.length > 0 && (
                <div className="space-y-2">
                  <Label>{t("providers.model_transformers")}</Label>
                  <div className="space-y-3">
                    {(editingProvider.models || []).map((model: string, modelIndex: number) => (
                      <div key={modelIndex} className="rounded-md border p-3">
                        <div className="mb-2 text-sm font-medium">{model}</div>
                        <div className="flex gap-2">
                          <div className="flex flex-1 gap-2">
                            <Combobox
                              options={availableTransformers.map((transformer) => ({
                                label: transformer.name,
                                value: transformer.name,
                              }))}
                              value=""
                              onChange={(value) => {
                                if (editingProviderIndex !== null) {
                                  handleModelTransformerChange(
                                    editingProviderIndex,
                                    model,
                                    value
                                  );
                                }
                              }}
                              placeholder={t("providers.select_transformer")}
                              emptyPlaceholder={t("providers.no_transformers")}
                            />
                          </div>
                        </div>

                        {editingProvider.transformer?.[model]?.use &&
                          editingProvider.transformer[model].use.length > 0 && (
                            <div className="mt-2 space-y-2">
                              <div className="text-sm font-medium text-gray-700">
                                {t("providers.selected_transformers")}
                              </div>
                              {editingProvider.transformer[model].use.map(
                                (
                                  transformer: string | (string | Record<string, unknown> | { max_tokens: number })[],
                                  transformerIndex: number
                                ) => (
                                  <div key={transformerIndex} className="rounded-md border p-3">
                                    <div className="mb-2 flex items-center gap-2">
                                      <div className="flex-1 rounded bg-gray-50 p-2 text-sm">
                                        {typeof transformer === "string"
                                          ? transformer
                                          : Array.isArray(transformer)
                                            ? String(transformer[0])
                                            : String(transformer)}
                                      </div>
                                      <Button
                                        variant="outline"
                                        size="icon"
                                        onClick={() => {
                                          if (editingProviderIndex !== null) {
                                            removeModelTransformerAtIndex(
                                              editingProviderIndex,
                                              model,
                                              transformerIndex
                                            );
                                          }
                                        }}
                                      >
                                        <Trash2 className="h-4 w-4" />
                                      </Button>
                                    </div>

                                    <div className="mt-2 border-l-2 border-gray-200 pl-4">
                                      <Label className="text-sm">
                                        {t("providers.transformer_parameters")}
                                      </Label>
                                      <div className="mt-1 space-y-2">
                                        <div className="flex gap-2">
                                          <Input
                                            placeholder={t("providers.parameter_name")}
                                            value={
                                              modelParamInputs[
                                                `model-${editingProviderIndex}-${model}-transformer-${transformerIndex}`
                                              ]?.name || ""
                                            }
                                            onChange={(e) => {
                                              const key = `model-${editingProviderIndex}-${model}-transformer-${transformerIndex}`;
                                              setModelParamInputs((prev) => ({
                                                ...prev,
                                                [key]: {
                                                  ...(prev[key] || {
                                                    name: "",
                                                    value: "",
                                                  }),
                                                  name: e.target.value,
                                                },
                                              }));
                                            }}
                                          />
                                          <Input
                                            placeholder={t("providers.parameter_value")}
                                            value={
                                              modelParamInputs[
                                                `model-${editingProviderIndex}-${model}-transformer-${transformerIndex}`
                                              ]?.value || ""
                                            }
                                            onChange={(e) => {
                                              const key = `model-${editingProviderIndex}-${model}-transformer-${transformerIndex}`;
                                              setModelParamInputs((prev) => ({
                                                ...prev,
                                                [key]: {
                                                  ...(prev[key] || {
                                                    name: "",
                                                    value: "",
                                                  }),
                                                  value: e.target.value,
                                                },
                                              }));
                                            }}
                                          />
                                          <Button
                                            size="sm"
                                            onClick={() => {
                                              if (editingProviderIndex !== null) {
                                                const key = `model-${editingProviderIndex}-${model}-transformer-${transformerIndex}`;
                                                const paramInput = modelParamInputs[key];
                                                if (
                                                  paramInput &&
                                                  paramInput.name &&
                                                  paramInput.value
                                                ) {
                                                  addModelTransformerParameter(
                                                    editingProviderIndex,
                                                    model,
                                                    transformerIndex,
                                                    paramInput.name,
                                                    paramInput.value
                                                  );
                                                  setModelParamInputs((prev) => ({
                                                    ...prev,
                                                    [key]: {
                                                      name: "",
                                                      value: "",
                                                    },
                                                  }));
                                                }
                                              }
                                            }}
                                          >
                                            <Plus className="h-4 w-4" />
                                          </Button>
                                        </div>

                                        {(() => {
                                          if (
                                            !editingProvider.transformer?.[model]?.use ||
                                            editingProvider.transformer[model].use.length <= transformerIndex
                                          ) {
                                            return null;
                                          }

                                          const targetTransformer =
                                            editingProvider.transformer[model].use[
                                              transformerIndex
                                            ];
                                          let params = {};

                                          if (
                                            Array.isArray(targetTransformer) &&
                                            targetTransformer.length > 1 &&
                                            typeof targetTransformer[1] === "object" &&
                                            targetTransformer[1] !== null
                                          ) {
                                            params = targetTransformer[1] as Record<string, unknown>;
                                          }

                                          return Object.keys(params).length > 0 ? (
                                            <div className="space-y-1">
                                              {Object.entries(params).map(([key, value]) => (
                                                <div
                                                  key={key}
                                                  className="flex items-center justify-between rounded bg-gray-50 p-2"
                                                >
                                                  <div className="text-sm">
                                                    <span className="font-medium">{key}:</span>{" "}
                                                    {String(value)}
                                                  </div>
                                                  <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    className="h-6 w-6 p-0"
                                                    onClick={() => {
                                                      if (editingProviderIndex !== null) {
                                                        removeModelTransformerParameterAtIndex(
                                                          editingProviderIndex,
                                                          model,
                                                          transformerIndex,
                                                          key
                                                        );
                                                      }
                                                    }}
                                                  >
                                                    <X className="h-3 w-3" />
                                                  </Button>
                                                </div>
                                              ))}
                                            </div>
                                          ) : null;
                                        })()}
                                      </div>
                                    </div>
                                  </div>
                                )
                              )}
                            </div>
                          )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          <div className="mt-auto space-y-3">
            <div className="flex justify-end gap-2">
              <Button onClick={handleSaveProvider}>{t("app.save")}</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deletingProviderIndex !== null}
        onOpenChange={() => setDeletingProviderIndex(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("providers.delete")}</DialogTitle>
            <DialogDescription>
              {t("providers.delete_provider_confirm")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeletingProviderIndex(null)}>
              {t("providers.cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={() =>
                deletingProviderIndex !== null &&
                handleRemoveProvider(deletingProviderIndex)
              }
            >
              {t("providers.delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
